"""
Inference pipeline for EVO-1 autonomous driving model

This module provides real-time inference capabilities for the EVO-1 model
with support for batch processing, streaming, and evaluation.
"""

import os
import time
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2
from PIL import Image
import asyncio
import websockets
import threading
from dataclasses import dataclass
import yaml

# Import EVO-1 components
from ..model.evo1_driving import EVO1Driving, FederatedEVO1Driving
from ..utils.config import EVO1DrivingConfig
from ..evaluation.driving_metrics import DrivingMetricsEvaluator
from ..data.nuscenes_loader import create_dataloader


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline"""
    model_path: str
    config_path: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 1
    max_sequence_length: int = 10
    confidence_threshold: float = 0.5
    enable_streaming: bool = False
    streaming_port: int = 8765
    save_predictions: bool = False
    output_dir: str = "./inference_outputs"
    use_federated_model: bool = False
    enable_profiling: bool = False


class DrivingInferencePipeline:
    """Main inference pipeline for EVO-1 autonomous driving"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load model and configuration
        self.load_model()
        
        # Setup data preprocessing
        self.setup_preprocessing()
        
        # Setup streaming if enabled
        if config.enable_streaming:
            self.setup_streaming()
        
        # Setup evaluator
        self.evaluator = None
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.output_dir, 'inference.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load trained EVO-1 model"""
        self.logger.info(f"Loading model from {self.config.model_path}")
        
        # Load configuration
        if self.config.config_path and os.path.exists(self.config.config_path):
            self.model_config = EVO1DrivingConfig.from_yaml(self.config.config_path)
        else:
            # Try to load config from model checkpoint
            try:
                checkpoint = torch.load(self.config.model_path, map_location=self.device)
                self.model_config = checkpoint.get('config')
                if self.model_config is None:
                    self.model_config = EVO1DrivingConfig()
            except Exception as e:
                self.logger.warning(f"Could not load config from checkpoint: {e}")
                self.model_config = EVO1DrivingConfig()
        
        # Load model
        if self.config.use_federated_model:
            self.model = FederatedEVO1Driving(
                config=self.model_config.model,
                device=self.device
            )
        else:
            self.model = EVO1Driving(
                config=self.model_config.model,
                training_config=self.model_config.training,
                device=self.device
            )
        
        # Load checkpoint
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
    
    def setup_preprocessing(self):
        """Setup data preprocessing pipeline"""
        # Image preprocessing
        import torchvision.transforms as transforms
        
        self.image_transform = transforms.Compose([
            transforms.Resize((self.model_config.data.image_size[0], 
                            self.model_config.data.image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.model_config.data.normalize_mean,
                std=self.model_config.data.normalize_std
            )
        ])
        
        # State normalization parameters
        self.state_mean = torch.zeros(12, device=self.device)
        self.state_std = torch.ones(12, device=self.device)
        
        # Control denormalization parameters
        self.control_ranges = {
            'steering': (-self.model_config.data.max_steering, self.model_config.data.max_steering),
            'throttle': (0.0, 1.0),
            'brake': (0.0, 1.0)
        }
    
    def setup_streaming(self):
        """Setup WebSocket streaming for real-time inference"""
        self.websocket_server = None
        self.streaming_clients = set()
        
        async def handle_client(websocket, path):
            self.streaming_clients.add(websocket)
            self.logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    await self.process_streaming_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.streaming_clients.remove(websocket)
                self.logger.info(f"Client disconnected: {websocket.remote_address}")
        
        # Start WebSocket server in separate thread
        def start_server():
            asyncio.set_event_loop(asyncio.new_event_loop())
            server = websockets.serve(handle_client, "localhost", self.config.streaming_port)
            asyncio.get_event_loop().run_until_complete(server)
            asyncio.get_event_loop().run_forever()
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        self.logger.info(f"WebSocket server started on port {self.config.streaming_port}")
    
    async def process_streaming_message(self, websocket, message):
        """Process incoming WebSocket message for inference"""
        try:
            # Parse message
            data = json.loads(message)
            
            # Run inference
            result = await self.inference_from_dict(data)
            
            # Send response
            response = {
                'timestamp': time.time(),
                'waypoints': result['waypoints'].tolist(),
                'controls': result['controls'].tolist(),
                'confidence': result['confidence'].item(),
                'inference_time': result['inference_time']
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            self.logger.error(f"Error processing streaming message: {e}")
            error_response = {'error': str(e)}
            await websocket.send(json.dumps(error_response))
    
    def preprocess_images(self, images: List[Union[str, np.ndarray, Image.Image]]) -> torch.Tensor:
        """Preprocess input images"""
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                # Load from file path
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                # Convert from numpy array
                img = Image.fromarray(img).convert('RGB')
            elif isinstance(img, Image.Image):
                img = img.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Apply transforms
            img_tensor = self.image_transform(img)
            processed_images.append(img_tensor)
        
        # Stack images [N, C, H, W]
        return torch.stack(processed_images)
    
    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess vehicle state"""
        state_tensor = torch.from_numpy(state).float().to(self.device)
        
        # Normalize (simple normalization for now)
        state_tensor = (state_tensor - self.state_mean) / (self.state_std + 1e-8)
        
        return state_tensor
    
    def denormalize_controls(self, controls: torch.Tensor) -> np.ndarray:
        """Denormalize control outputs to real values"""
        controls_np = controls.cpu().numpy()
        
        # Denormalize each control channel
        denormalized = np.zeros_like(controls_np)
        
        # Steering [-1, 1] -> [min_steering, max_steering]
        denormalized[..., 0] = ((controls_np[..., 0] + 1) / 2) * (
            self.control_ranges['steering'][1] - self.control_ranges['steering'][0]
        ) + self.control_ranges['steering'][0]
        
        # Throttle [-1, 1] -> [0, 1]
        denormalized[..., 1] = ((controls_np[..., 1] + 1) / 2)
        
        # Brake [-1, 1] -> [0, 1]
        denormalized[..., 2] = ((controls_np[..., 2] + 1) / 2)
        
        return denormalized
    
    @torch.no_grad()
    def inference(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        state: np.ndarray,
        instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run inference on single sample"""
        start_time = time.time()
        
        # Preprocess inputs
        images_tensor = self.preprocess_images(images)  # [N, C, H, W]
        state_tensor = self.preprocess_state(state)     # [12]
        
        # Add batch dimension
        images_tensor = images_tensor.unsqueeze(0)  # [1, N, C, H, W]
        state_tensor = state_tensor.unsqueeze(0)   # [1, 12]
        
        # Create image mask (all cameras available)
        image_mask = torch.ones(1, images_tensor.shape[1], device=self.device)
        
        # Prepare instruction
        instructions = [instruction] if instruction else ["Continue driving safely"]
        
        # Run inference
        output = self.model(
            images=images_tensor,
            image_mask=image_mask,
            state=state_tensor,
            instructions=instructions,
            mode="inference"
        )
        
        # Post-process outputs
        waypoints = output.waypoints.squeeze(0)      # [T, 3]
        controls = output.controls.squeeze(0)        # [T, 3]
        confidence = output.confidence.squeeze(0)     # [1]
        
        # Denormalize controls
        denormalized_controls = self.denormalize_controls(controls)
        
        # Filter low confidence predictions
        if confidence.item() < self.config.confidence_threshold:
            self.logger.warning(f"Low confidence prediction: {confidence.item():.3f}")
        
        # Compute inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        return {
            'waypoints': waypoints.cpu(),
            'controls': torch.from_numpy(denormalized_controls),
            'confidence': confidence.cpu(),
            'inference_time': inference_time,
            'raw_controls': controls.cpu(),
            'vision_features': output.vision_features.cpu()
        }
    
    async def inference_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference from dictionary input (for streaming)"""
        # Extract images
        images = []
        if 'images' in data:
            for img_data in data['images']:
                if isinstance(img_data, str):
                    # Base64 encoded image or file path
                    if img_data.startswith('data:image'):
                        # Handle base64 encoded image
                        import base64
                        from io import BytesIO
                        img_data = base64.b64decode(img_data.split(',')[1])
                        img = Image.open(BytesIO(img_data))
                    else:
                        # File path
                        img = Image.open(img_data)
                    images.append(img)
                elif isinstance(img_data, list):
                    # Raw pixel data [H, W, C]
                    img_array = np.array(img_data, dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    images.append(img)
        
        # Extract state
        state = np.array(data.get('state', np.zeros(12)), dtype=np.float32)
        
        # Extract instruction
        instruction = data.get('instruction', 'Continue driving safely')
        
        # Run inference
        return self.inference(images, state, instruction)
    
    def batch_inference(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run inference on batch of samples"""
        results = []
        
        # Process in mini-batches for memory efficiency
        batch_size = min(self.config.batch_size, len(batch_data))
        
        for i in range(0, len(batch_data), batch_size):
            mini_batch = batch_data[i:i + batch_size]
            
            # Prepare batch tensors
            batch_images = []
            batch_states = []
            batch_instructions = []
            
            for data in mini_batch:
                # Images
                images = []
                for img in data['images']:
                    if isinstance(img, str):
                        images.append(Image.open(img).convert('RGB'))
                    elif isinstance(img, np.ndarray):
                        images.append(Image.fromarray(img).convert('RGB'))
                    else:
                        images.append(img)
                batch_images.append(images)
                
                # State
                batch_states.append(np.array(data['state'], dtype=np.float32))
                
                # Instruction
                batch_instructions.append(data.get('instruction', 'Continue driving safely'))
            
            # Run batch inference
            try:
                batch_results = self._batch_inference_internal(batch_images, batch_states, batch_instructions)
                results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Batch inference failed: {e}")
                # Fallback to individual inference
                for data in mini_batch:
                    try:
                        result = self.inference(data['images'], np.array(data['state']), 
                                              data.get('instruction'))
                        results.append(result)
                    except Exception as e2:
                        self.logger.error(f"Individual inference failed: {e2}")
                        results.append(None)
        
        return results
    
    def _batch_inference_internal(
        self,
        batch_images: List[List[Image.Image]],
        batch_states: List[np.ndarray],
        batch_instructions: List[str]
    ) -> List[Dict[str, Any]]:
        """Internal batch inference implementation"""
        batch_size = len(batch_images)
        num_views = len(batch_images[0])
        
        # Prepare batch tensors
        images_batch = torch.zeros(batch_size, num_views, 3, 
                                 self.model_config.data.image_size[0], 
                                 self.model_config.data.image_size[1], device=self.device)
        states_batch = torch.zeros(batch_size, 12, device=self.device)
        
        # Process each sample
        for i in range(batch_size):
            # Images
            for j, img in enumerate(batch_images[i]):
                img_tensor = self.image_transform(img)
                images_batch[i, j] = img_tensor
            
            # State
            state_tensor = torch.from_numpy(batch_states[i]).float().to(self.device)
            state_tensor = (state_tensor - self.state_mean) / (self.state_std + 1e-8)
            states_batch[i] = state_tensor
        
        # Create image mask
        image_mask = torch.ones(batch_size, num_views, device=self.device)
        
        # Run model forward pass
        output = self.model(
            images=images_batch,
            image_mask=image_mask,
            state=states_batch,
            instructions=batch_instructions,
            mode="inference"
        )
        
        # Process outputs
        results = []
        for i in range(batch_size):
            waypoints = output.waypoints[i].cpu()
            controls = output.controls[i].cpu()
            confidence = output.confidence[i].cpu()
            
            # Denormalize controls
            denormalized_controls = self.denormalize_controls(controls.unsqueeze(0))[0]
            
            results.append({
                'waypoints': waypoints,
                'controls': torch.from_numpy(denormalized_controls),
                'confidence': confidence,
                'inference_time': 0.0,  # Not measured per sample in batch
                'raw_controls': controls
            })
        
        return results
    
    def evaluate_model(
        self,
        test_loader: torch.utils.data.DataLoader,
        evaluator: Optional[DrivingMetricsEvaluator] = None
    ) -> Dict[str, Any]:
        """Evaluate model on test dataset"""
        self.logger.info("Starting model evaluation...")
        
        if evaluator is None:
            evaluator = DrivingMetricsEvaluator(
                config=self.model_config.evaluation,
                model_config=self.model_config.model,
                output_dir=self.config.output_dir
            )
        
        all_results = []
        total_samples = 0
        total_time = 0.0
        
        for batch_idx, batch in enumerate(test_loader):
            self.logger.info(f"Evaluating batch {batch_idx + 1}/{len(test_loader)}")
            
            batch_start_time = time.time()
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Run inference
            with torch.no_grad():
                output = self.model(
                    images=batch['images'],
                    image_mask=batch['image_mask'],
                    state=batch['state'],
                    instructions=batch['instructions'],
                    mode="inference"
                )
            
            # Evaluate batch
            batch_results = evaluator.evaluate_batch(batch, output)
            all_results.append(batch_results)
            
            batch_time = time.time() - batch_start_time
            total_time += batch_time
            total_samples += len(batch['images'])
            
            # Log progress
            if batch_idx % 10 == 0:
                avg_time = batch_time / len(batch['images'])
                self.logger.info(f"Batch {batch_idx + 1}: {avg_time:.4f}s per sample")
        
        # Compute overall metrics
        overall_metrics = self._compute_overall_metrics(all_results)
        overall_metrics['total_inference_time'] = total_time
        overall_metrics['samples_processed'] = total_samples
        overall_metrics['avg_inference_time_per_sample'] = total_time / total_samples
        overall_metrics['throughput'] = total_samples / total_time
        
        # Save evaluation results
        self.save_evaluation_results(overall_metrics)
        
        # Visualize results
        evaluator.visualize_results(
            save_path=os.path.join(self.config.output_dir, 'evaluation_visualization.png')
        )
        
        self.logger.info("Evaluation completed")
        return overall_metrics
    
    def _compute_overall_metrics(self, all_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall metrics from batch results"""
        overall = {}
        
        if not all_results:
            return overall
        
        # Aggregate trajectory metrics
        trajectory_metrics = []
        control_metrics = []
        safety_metrics = []
        efficiency_metrics = []
        
        for result in all_results:
            if 'trajectory' in result:
                trajectory_metrics.append(result['trajectory'])
            if 'control' in result:
                control_metrics.append(result['control'])
            if 'safety' in result:
                safety_metrics.append(result['safety'])
            if 'efficiency' in result:
                efficiency_metrics.append(result['efficiency'])
        
        # Compute averages
        if trajectory_metrics:
            overall['avg_ade'] = np.mean([m.ade for m in trajectory_metrics])
            overall['avg_fde'] = np.mean([m.fde for m in trajectory_metrics])
            overall['avg_miss_rate'] = np.mean([m.miss_rate for m in trajectory_metrics])
        
        if control_metrics:
            overall['avg_steering_rmse'] = np.mean([m.steering_rmse for m in control_metrics])
            overall['avg_throttle_rmse'] = np.mean([m.throttle_rmse for m in control_metrics])
            overall['avg_brake_rmse'] = np.mean([m.brake_rmse for m in control_metrics])
        
        if safety_metrics:
            overall['avg_collision_rate'] = np.mean([m.collision_rate for m in safety_metrics])
            overall['avg_comfort_score'] = np.mean([m.comfort_score for m in safety_metrics])
        
        if efficiency_metrics:
            overall['avg_fuel_efficiency'] = np.mean([m.fuel_efficiency_score for m in efficiency_metrics])
        
        return overall
    
    def save_evaluation_results(self, metrics: Dict[str, Any]):
        """Save evaluation results to file"""
        results_path = os.path.join(self.config.output_dir, 'evaluation_results.json')
        
        # Convert tensors to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {results_path}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_frames_processed': self.frame_count,
            'avg_fps': self.frame_count / sum(self.inference_times) if sum(self.inference_times) > 0 else 0
        }
    
    def save_performance_stats(self):
        """Save performance statistics"""
        stats = self.get_performance_stats()
        
        if stats:
            stats_path = os.path.join(self.config.output_dir, 'performance_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"Performance stats saved to {stats_path}")
    
    def save_predictions(self, predictions: List[Dict[str, Any]], filename: str = "predictions.json"):
        """Save predictions to file"""
        if not self.config.save_predictions:
            return
        
        predictions_path = os.path.join(self.config.output_dir, filename)
        
        # Convert tensors to lists for JSON serialization
        serializable_predictions = []
        for pred in predictions:
            if pred is None:
                continue
            
            serializable_pred = {}
            for key, value in pred.items():
                if isinstance(value, torch.Tensor):
                    serializable_pred[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    serializable_pred[key] = value.tolist()
                else:
                    serializable_pred[key] = value
            
            serializable_predictions.append(serializable_pred)
        
        with open(predictions_path, 'w') as f:
            json.dump(serializable_predictions, f, indent=2)
        
        self.logger.info(f"Predictions saved to {predictions_path}")
    
    def __del__(self):
        """Cleanup when pipeline is destroyed"""
        if hasattr(self, 'inference_times') and self.inference_times:
            self.save_performance_stats()


# Utility functions
def create_inference_config(args) -> InferenceConfig:
    """Create inference configuration from arguments"""
    return InferenceConfig(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        confidence_threshold=args.confidence_threshold,
        enable_streaming=args.enable_streaming,
        streaming_port=args.streaming_port,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
        use_federated_model=args.use_federated_model,
        enable_profiling=args.enable_profiling
    )


def parse_inference_arguments():
    """Parse command line arguments for inference"""
    parser = argparse.ArgumentParser(
        description="EVO-1 Autonomous Driving Inference"
    )
    
    # Model
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config_path', type=str, default=None,
        help='Path to model configuration file'
    )
    
    # Inference settings
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--max_sequence_length', type=int, default=10,
        help='Maximum sequence length for processing'
    )
    parser.add_argument(
        '--confidence_threshold', type=float, default=0.5,
        help='Confidence threshold for predictions'
    )
    
    # Streaming
    parser.add_argument(
        '--enable_streaming', action='store_true',
        help='Enable WebSocket streaming for real-time inference'
    )
    parser.add_argument(
        '--streaming_port', type=int, default=8765,
        help='Port for WebSocket streaming server'
    )
    
    # Output
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Save predictions to file'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./inference_outputs',
        help='Output directory for results'
    )
    
    # Model type
    parser.add_argument(
        '--use_federated_model', action='store_true',
        help='Use federated model architecture'
    )
    
    # Profiling
    parser.add_argument(
        '--enable_profiling', action='store_true',
        help='Enable performance profiling'
    )
    
    # Evaluation
    parser.add_argument(
        '--evaluate', action='store_true',
        help='Run evaluation on test set'
    )
    parser.add_argument(
        '--data_root', type=str, default='/data/nuscenes',
        help='Root directory for test data'
    )
    
    return parser.parse_args()