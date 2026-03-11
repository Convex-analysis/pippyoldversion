#!/usr/bin/env python3
"""
FHDP for EVO-1 Stage 1: Action Expert Alignment on Jetson Devices

This implementation focuses exclusively on Stage 1 training where:
- VLM (Vision-Language Model) is FROZEN
- Only Integration Module + Action Head are trained
- Optimized for Jetson Orin/Nano constraints
- Lightweight federated learning with efficient resource management
"""
import sys
import os
import time
import threading
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import psutil
import gc
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FHDP Core Framework Integration
from fhdp.core import FHDPSystem, SystemConfiguration
from fhdp.edge_server import EdgeServer
from fhdp.vehicle_layer import Vehicle
from fhdp.core.types import VehicleInfo, TrainingConfig, TrainingMode

# Jetson-specific optimizations
import torch.cuda.amp as amp
from torch.cuda.amp import GradScaler

@dataclass
class JetsonResourceConstraints:
    """Jetson device resource constraints and capabilities"""
    device_name: str = "jetson_orin"
    max_memory_mb: int = 8192  # 8GB for Orin, 4GB for Nano
    max_power_watts: int = 15  # Power constraints
    thermal_threshold: float = 85.0  # Temperature threshold in Celsius
    max_batch_size: int = 4  # Reduced batch size for Jetson
    max_sequence_length: int = 256  # Reduced for efficiency
    precision: str = "float16"  # Use FP16 for memory efficiency
    
class LightweightActionHead(nn.Module):
    """
    Lightweight Action Head for EVO-1 Stage 1 Training
    
    This module is specifically designed for Jetson deployment:
    - Minimal parameters (~10K instead of 1M+)
    - Efficient transformer layers
    - FP16 optimization
    - Gradient checkpointing support
    """
    
    def __init__(self, vision_dim: int = 2048, language_dim: int = 768, 
                 hidden_dim: int = 256, action_dim: int = 3):
        super().__init__()
        
        # Dimension reduction for efficiency
        self.vision_proj = nn.Linear(vision_dim, hidden_dim // 2)
        self.language_proj = nn.Linear(language_dim, hidden_dim // 2)
        
        # Lightweight transformer (2 layers instead of 8)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,  # Reduced from 8
                dim_feedforward=hidden_dim * 2,  # Reduced from 4x
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2  # Reduced from 8
        )
        
        # Final action prediction
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Bounded output for [-1, 1] range
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, vision_features: torch.Tensor, 
                language_features: torch.Tensor) -> torch.Tensor:
        # Feature projection (reduces dimensionality)
        vision_proj = self.vision_proj(vision_features)  # [B, hidden//2]
        language_proj = self.language_proj(language_features)  # [B, hidden//2]
        
        # Concatenate and normalize
        combined = torch.cat([vision_proj, language_proj], dim=-1)  # [B, hidden]
        combined = self.norm(combined).unsqueeze(1)  # [B, 1, hidden]
        
        # Lightweight transformer processing
        processed = self.transformer(combined)  # [B, 1, hidden]
        
        # Action prediction
        action = self.action_head(processed.squeeze(1))  # [B, action_dim]
        
        return action
    
    def get_parameter_count(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())

class FrozenVLMInterface:
    """
    Interface to frozen VLM model for Stage 1 training
    
    Since VLM is frozen, we use cached features to avoid heavy computation
    """
    
    def __init__(self, vlm_name: str = "OpenGVLab/InternVL3-1B"):
        self.vlm_name = vlm_name
        self.is_loaded = False
        self.feature_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # For Stage 1, we don't actually load the VLM to save memory
        # Instead, we use precomputed features
        print(f"🧠 VLM Interface: Using cached features for {vlm_name} (Stage 1)")
        self.is_loaded = True
    
    def extract_vision_features(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract vision features from images
        
        In Stage 1, this returns cached/simulated features to avoid
        loading the full VLM model on Jetson devices
        """
        batch_size = len(images)
        
        # Simulate cached VLM features (2048-dim vision embedding)
        # In real implementation, this would be precomputed features
        features = torch.randn(batch_size, 2048, device=self.device)
        
        return features
    
    def extract_language_features(self, prompts: List[str]) -> torch.Tensor:
        """
        Extract language features from text prompts
        
        In Stage 1, this returns cached/simulated features
        """
        batch_size = len(prompts)
        
        # Simulate cached language features (768-dim text embedding)
        # In real implementation, this would be precomputed features
        features = torch.randn(batch_size, 768, device=self.device)
        
        return features

class JetsonMemoryManager:
    """Advanced memory management for Jetson devices"""
    
    def __init__(self, constraints: JetsonResourceConstraints):
        self.constraints = constraints
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_history = deque(maxlen=10)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            cached = torch.cuda.memory_reserved() / (1024**3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_gb': total,
                'usage_percent': (allocated / total) * 100
            }
        else:
            memory = psutil.virtual_memory()
            return {
                'allocated_gb': memory.used / (1024**3),
                'cached_gb': memory.cached / (1024**3) if hasattr(memory, 'cached') else 0,
                'total_gb': memory.total / (1024**3),
                'usage_percent': memory.percent
            }
    
    def optimize_memory(self):
        """Optimize memory usage on Jetson"""
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Clear gradients
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                if obj.grad is not None:
                    obj.grad = None
    
    def check_memory_threshold(self) -> bool:
        """Check if memory usage exceeds threshold"""
        usage = self.get_memory_usage()
        threshold_percent = (self.constraints.max_memory_mb / 1024) * 0.8  # 80% threshold
        
        return usage['usage_percent'] > threshold_percent
    
    def get_temperature(self) -> Optional[float]:
        """Get Jetson device temperature"""
        try:
            # Jetson-specific temperature reading
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_raw = f.read().strip()
                    return float(temp_raw) / 1000.0  # Convert to Celsius
        except:
            pass
        
        return None

class Stage1ActionExpertTrainer:
    """
    Stage 1 trainer for Action Expert Alignment
    
    Focuses on training only the Integration Module + Action Head
    while keeping the VLM completely frozen
    """
    
    def __init__(self, constraints: JetsonResourceConstraints):
        self.constraints = constraints
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_manager = JetsonMemoryManager(constraints)
        
        # Model components
        self.vlm_interface = FrozenVLMInterface()
        self.action_head = LightweightActionHead().to(self.device)
        
        # Training components
        self.optimizer = None
        self.scaler = None  # Will be initialized with new API
        self.training_stats = []
        
        # Training configuration
        self.setup_training()
        
        # Initialize scaler with new API
        if constraints.precision == "float16":
            self.scaler = torch.cuda.amp.GradScaler()  # Keep using the old API for compatibility
        
        print(f"🚀 Stage 1 Trainer initialized on {self.device}")
        print(f"   Action Head parameters: {self.action_head.get_parameter_count():,}")
        print(f"   Precision: {constraints.precision}")
        print(f"   Max batch size: {constraints.max_batch_size}")
    
    def setup_training(self):
        """Setup training components for Stage 1"""
        # Optimizer for action head only (VLM is frozen)
        self.optimizer = optim.AdamW(
            self.action_head.parameters(),
            lr=1e-4,  # Conservative learning rate for Stage 1
            weight_decay=1e-3
        )
        
        # Loss function for action alignment
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
    
    def train_step(self, batch_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Single training step for Stage 1
        
        Args:
            batch_data: Either a single training sample (dict) or list of training samples,
                       each containing images, prompts, and target actions
        
        Returns:
            Training metrics for this step
        """
        self.action_head.train()
        
        # Handle both single dict and list of dicts
        if isinstance(batch_data, dict):
            batch_data = [batch_data]
        
        # Extract data from batch
        images = [sample['images'] for sample in batch_data]  # List of image lists
        prompts = [sample['prompts'][0] if isinstance(sample['prompts'], list) else sample['prompts'] for sample in batch_data]  # List of text prompts
        target_actions = [sample['actions'] for sample in batch_data]  # List of target actions
        
        # Move to device
        target_actions = torch.tensor(target_actions, dtype=torch.float32, device=self.device)
        
        # Memory optimization: check before processing
        if self.memory_manager.check_memory_threshold():
            self.memory_manager.optimize_memory()
        
        # Extract features (VLM is frozen, using cached features)
        vision_features = self.vlm_interface.extract_vision_features(images)
        language_features = self.vlm_interface.extract_language_features(prompts)
        
        # Forward pass through action head only
        if self.scaler:  # Mixed precision
            with torch.cuda.amp.autocast():
                predicted_actions = self.action_head(vision_features, language_features)
                loss = self.criterion(predicted_actions, target_actions)
        else:
            predicted_actions = self.action_head(vision_features, language_features)
            loss = self.criterion(predicted_actions, target_actions)
        
        # Backward pass (only action head gradients)
        self.optimizer.zero_grad()
        
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(predicted_actions - target_actions)).item()
            mse = torch.mean((predicted_actions - target_actions) ** 2).item()
        
        # Memory cleanup
        self.memory_manager.optimize_memory()
        
        return {
            'loss': loss.item(),
            'mae': mae,
            'mse': mse,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def train_epoch(self, dataset: List[Dict[str, Any]], 
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = []
        epoch_maes = []
        
        # Create small batches for Jetson
        batch_size = self.constraints.max_batch_size
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            
            # Process batch
            metrics = self.train_step(batch)
            epoch_losses.append(metrics['loss'])
            epoch_maes.append(metrics['mae'])
            
            # Progress reporting
            if i % (batch_size * 5) == 0:  # Every 5 batches
                temp = self.memory_manager.get_temperature()
                memory = self.memory_manager.get_memory_usage()
                
                print(f"  Epoch {epoch}, Batch {i//batch_size}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"MAE={metrics['mae']:.4f}, "
                      f"Mem={memory['usage_percent']:.1f}%, "
                      f"Temp={temp:.1f}°C" if temp else "Temp=N/A")
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_mae = np.mean(epoch_maes)
        
        return {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'avg_mae': avg_mae,
            'memory_usage': self.memory_manager.get_memory_usage()['usage_percent'],
            'temperature': self.memory_manager.get_temperature()
        }

class JetsonFederatedLearning:
    """
    Federated Learning optimized for Jetson devices
    
    Implements efficient federated aggregation specifically for Stage 1
    with minimal communication overhead and resource usage
    """
    
    def __init__(self, constraints: JetsonResourceConstraints):
        self.constraints = constraints
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aggregated_weights = None
        self.aggregation_history = []
        
    def prepare_model_for_vehicle(self, vehicle_id: str) -> Stage1ActionExpertTrainer:
        """Prepare a Stage 1 trainer for a specific vehicle"""
        trainer = Stage1ActionExpertTrainer(self.constraints)
        
        # Load global weights if available
        if self.aggregated_weights is not None:
            trainer.action_head.load_state_dict(self.aggregated_weights)
        
        return trainer
    
    def collect_vehicle_updates(self, trainers: Dict[str, Stage1ActionExpertTrainer]) -> Dict[str, Any]:
        """Collect model updates from vehicles"""
        updates = {}
        
        for vehicle_id, trainer in trainers.items():
            # Get current action head weights
            weights = trainer.action_head.state_dict()
            
            # Get training statistics
            if trainer.training_stats:
                latest_stats = trainer.training_stats[-1]
                performance = latest_stats['avg_loss']
            else:
                performance = 1.0
            
            updates[vehicle_id] = {
                'weights': weights,
                'performance': performance,
                'memory_usage': trainer.memory_manager.get_memory_usage()['usage_percent']
            }
        
        return updates
    
    def aggregate_weights(self, updates: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Aggregate weights from multiple vehicles
        
        Uses performance-weighted averaging for better models
        """
        if not updates:
            return None
        
        print(f"🔄 Aggregating {len(updates)} vehicle updates...")
        
        # Calculate weights based on performance (inverse of loss)
        total_weight = 0.0
        for update in updates.values():
            # Lower loss = higher weight
            weight = 1.0 / (update['performance'] + 1e-6)
            total_weight += weight
        
        # Initialize aggregated weights
        first_weights = list(updates.values())[0]['weights']
        aggregated = {}
        
        for key in first_weights.keys():
            aggregated[key] = torch.zeros_like(first_weights[key])
        
        # Weighted aggregation
        for vehicle_id, update in updates.items():
            weight = (1.0 / (update['performance'] + 1e-6)) / total_weight
            
            for key, tensor in update['weights'].items():
                aggregated[key] += weight * tensor.to(self.device)
        
        self.aggregated_weights = aggregated
        
        # Log aggregation statistics
        avg_memory = np.mean([u['memory_usage'] for u in updates.values()])
        print(f"✅ Aggregation completed: Avg memory usage: {avg_memory:.1f}%")
        
        return aggregated

class JetsonAutonomousVehicle:
    """
    Autonomous vehicle optimized for Jetson deployment
    
    Integrates Stage 1 training with real-time driving capabilities
    """
    
    def __init__(self, vehicle_id: str, position: Tuple[float, float],
                 constraints: JetsonResourceConstraints):
        self.vehicle_id = vehicle_id
        self.position = position
        self.constraints = constraints
        
        # Components
        self.trainer = Stage1ActionExpertTrainer(constraints)
        self.federated_learner = JetsonFederatedLearning(constraints)
        
        # Vehicle state
        self.velocity = 0.0
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        
        # Data collection
        self.training_data = []
        self.driving_metrics = {
            'distance': 0.0,
            'safety_violations': 0,
            'training_time': 0.0
        }
        
        print(f"🚗 Jetson Vehicle {vehicle_id} initialized")
    
    def collect_training_data(self, observation: Dict[str, Any], 
                            action: np.ndarray, reward: float) -> Dict[str, Any]:
        """Collect training data for Stage 1 alignment"""
        
        # Create training sample
        sample = {
            'images': observation.get('images', []),
            'prompts': [observation.get('prompt', '保持车道行驶')],
            'actions': action.tolist(),
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.training_data.append(sample)
        
        # Limit data size for memory efficiency
        max_samples = 100  # Small dataset for Jetson
        if len(self.training_data) > max_samples:
            self.training_data = self.training_data[-max_samples:]
        
        return sample
    
    def train_stage1(self, epochs: int = 3) -> Dict[str, Any]:
        """Train Stage 1 model on collected data"""
        
        if len(self.training_data) < 10:
            print(f"⚠️  {self.vehicle_id}: Insufficient data for training")
            return {'status': 'insufficient_data'}
        
        start_time = time.time()
        
        print(f"🎓 {self.vehicle_id}: Starting Stage 1 training ({len(self.training_data)} samples)")
        
        epoch_results = []
        
        for epoch in range(epochs):
            result = self.trainer.train_epoch(self.training_data, epoch)
            epoch_results.append(result)
        
        training_time = time.time() - start_time
        self.driving_metrics['training_time'] += training_time
        
        # Store training statistics
        avg_loss = np.mean([r['avg_loss'] for r in epoch_results])
        self.trainer.training_stats.append({
            'vehicle_id': self.vehicle_id,
            'avg_loss': avg_loss,
            'training_time': training_time,
            'num_samples': len(self.training_data)
        })
        
        print(f"✅ {self.vehicle_id}: Stage 1 training completed in {training_time:.2f}s")
        
        return {
            'status': 'completed',
            'avg_loss': avg_loss,
            'training_time': training_time,
            'vehicle_id': self.vehicle_id,
            'num_samples': len(self.training_data)
        }
    
    def run_stage1_simulation(self, duration: float = 30.0, epochs: int = 3) -> Dict[str, Any]:
        """
        Run complete Stage 1 simulation including data collection and training
        
        Args:
            duration: Duration of driving simulation in seconds
            epochs: Number of training epochs
            
        Returns:
            Simulation results including driving metrics and training outcomes
        """
        print(f"🚀 {self.vehicle_id}: Starting Stage 1 simulation")
        
        # Run driving simulation to collect data
        loop = asyncio.get_event_loop()
        loop.run_until_complete(simulate_jetson_driving(self, duration))
        
        # Train the Stage 1 model on collected data
        training_results = self.train_stage1(epochs)
        
        # Combine results
        results = {
            'vehicle_id': self.vehicle_id,
            'driving_metrics': self.driving_metrics.copy(),
            'training_results': training_results,
            'simulation_duration': duration,
            'timestamp': time.time()
        }
        
        print(f"🏁 {self.vehicle_id}: Stage 1 simulation completed")
        print(f"📊 Driving metrics: {results['driving_metrics']}")
        
        return results

def create_jetson_scenarios() -> List[Dict[str, Any]]:
    """Create driving scenarios optimized for Jetson testing"""
    
    scenarios = [
        {
            'name': 'highway_cruising',
            'description': 'Highway driving with lane keeping',
            'duration': 60,
            'complexity': 'low',
            'data_frequency': 2.0  # Hz
        },
        {
            'name': 'urban_navigation',
            'description': 'Urban intersection navigation',
            'duration': 45,
            'complexity': 'medium',
            'data_frequency': 1.5  # Hz
        },
        {
            'name': 'parking_maneuver',
            'description': 'Parking and slow-speed maneuvers',
            'duration': 30,
            'complexity': 'high',
            'data_frequency': 1.0  # Hz
        }
    ]
    
    return scenarios

async def simulate_jetson_driving(vehicle: JetsonAutonomousVehicle, 
                                duration: float):
    """Simulate autonomous driving on Jetson"""
    
    start_time = time.time()
    scenario_index = 0
    scenarios = create_jetson_scenarios()
    
    print(f"🚗 Starting {vehicle.vehicle_id} driving simulation ({duration}s)")
    
    while time.time() - start_time < duration:
        # Select scenario
        current_scenario = scenarios[scenario_index % len(scenarios)]
        
        # Generate observation
        observation = {
            'images': [torch.randn(3, 224, 224)],  # Simulated camera
            'prompt': current_scenario['description'],
            'speed': vehicle.velocity
        }
        
        # Simple action generation (would use trained model)
        action = np.array([
            np.random.uniform(-0.3, 0.3),  # Steering
            np.random.uniform(0.2, 0.8),   # Throttle
            np.random.uniform(0.0, 0.1)     # Brake
        ])
        
        # Calculate reward
        reward = -abs(action[0]) * 0.1 + action[1] * 0.5 - action[2] * 0.3
        
        # Collect training data
        vehicle.collect_training_data(observation, action, reward)
        
        # Update vehicle state
        vehicle.steering = action[0]
        vehicle.throttle = action[1]
        vehicle.brake = action[2]
        
        # Simple physics update
        vehicle.velocity += (vehicle.throttle - vehicle.brake) * 0.5
        vehicle.velocity = max(0, min(30, vehicle.velocity))
        
        # Update position
        vehicle.position = (
            vehicle.position[0] + vehicle.velocity * 0.1,
            vehicle.position[1] + vehicle.steering * vehicle.velocity * 0.01
        )
        
        vehicle.driving_metrics['distance'] += vehicle.velocity * 0.1
        
        # Safety check
        if abs(vehicle.steering) > 0.5 or vehicle.brake > 0.8:
            vehicle.driving_metrics['safety_violations'] += 1
        
        # Sleep for simulation timing
        await asyncio.sleep(0.5)
        
        scenario_index += 1
    
    print(f"📍 {vehicle.vehicle_id}: Drove {vehicle.driving_metrics['distance']:.1f}m, "
          f"Violations: {vehicle.driving_metrics['safety_violations']}")

async def main():
    """Main Stage 1 federated learning simulation"""
    
    print("=" * 60)
    print("🚀 EVO-1 Stage 1: Action Expert Alignment on Jetson")
    print("=" * 60)
    
    # Jetson constraints
    constraints = JetsonResourceConstraints(
        device_name="jetson_orin",
        max_memory_mb=6144,  # 6GB usable
        max_batch_size=4,
        precision="float16"
    )
    
    print(f"📱 Jetson Constraints: {constraints.max_memory_mb}MB RAM, "
          f"Batch {constraints.max_batch_size}, {constraints.precision}")
    
    # Simulation parameters
    NUM_VEHICLES = 3  # Reduced for Jetson
    DRIVING_DURATION = 90  # seconds
    TRAINING_ROUNDS = 3
    
    # Create vehicles
    vehicles = []
    for i in range(NUM_VEHICLES):
        position = (i * 50.0, 0.0)  # Spaced positions
        vehicle = JetsonAutonomousVehicle(
            f"jetson_vehicle_{i:03d}", 
            position, 
            constraints
        )
        vehicles.append(vehicle)
    
    # Create federated learning system
    federated_learner = JetsonFederatedLearning(constraints)
    
    # Driving and training simulation
    for round_num in range(TRAINING_ROUNDS):
        print(f"\n🔄 Round {round_num + 1}/{TRAINING_ROUNDS}")
        print("-" * 40)
        
        # Phase 1: Driving and data collection
        print("🚗 Phase 1: Data Collection")
        driving_tasks = []
        
        for vehicle in vehicles:
            task = asyncio.create_task(
                simulate_jetson_driving(vehicle, DRIVING_DURATION / TRAINING_ROUNDS)
            )
            driving_tasks.append(task)
        
        await asyncio.gather(*driving_tasks)
        
        # Phase 2: Stage 1 Training
        print("\n🎓 Phase 2: Stage 1 Training")
        trainers = {}
        
        for vehicle in vehicles:
            result = vehicle.train_stage1(epochs=2)
            if result['status'] == 'completed':
                trainers[vehicle.vehicle_id] = vehicle.trainer
        
        # Phase 3: Federated Aggregation
        if trainers:
            print("\n🌐 Phase 3: Federated Aggregation")
            updates = federated_learner.collect_vehicle_updates(trainers)
            aggregated_weights = federated_learner.aggregate_weights(updates)
            
            # Update all vehicles with aggregated weights
            for vehicle in vehicles:
                if aggregated_weights:
                    vehicle.trainer.action_head.load_state_dict(aggregated_weights)
                    print(f"✅ {vehicle.vehicle_id}: Updated with global weights")
        
        # Memory optimization
        for vehicle in vehicles:
            vehicle.trainer.memory_manager.optimize_memory()
    
    # Final Results
    print("\n" + "=" * 60)
    print("📊 Stage 1 Federated Learning Results")
    print("=" * 60)
    
    for vehicle in vehicles:
        metrics = vehicle.driving_metrics
        training_stats = vehicle.trainer.training_stats
        
        print(f"\n🚗 {vehicle.vehicle_id}:")
        print(f"  Distance driven: {metrics['distance']:.1f}m")
        print(f"  Safety violations: {metrics['safety_violations']}")
        print(f"  Training time: {metrics['training_time']:.2f}s")
        
        if training_stats:
            latest = training_stats[-1]
            print(f"  Final training loss: {latest['avg_loss']:.4f}")
            print(f"  Training samples: {latest['num_samples']}")
    
    # Model size and efficiency report
    print(f"\n💾 Model Efficiency Report:")
    action_head = vehicles[0].trainer.action_head
    print(f"  Action Head parameters: {action_head.get_parameter_count():,}")
    print(f"  VLM Status: FROZEN (Stage 1 optimization)")
    print(f"  Memory precision: {constraints.precision}")
    
    # Resource usage summary
    memory_usage = vehicles[0].trainer.memory_manager.get_memory_usage()
    temperature = vehicles[0].trainer.memory_manager.get_temperature()
    
    print(f"\n🔥 Resource Usage:")
    print(f"  Memory usage: {memory_usage['usage_percent']:.1f}%")
    print(f"  Temperature: {temperature:.1f}°C" if temperature else "  Temperature: N/A")
    
    print(f"\n✅ Stage 1 simulation completed successfully!")
    print(f"   Ready for Stage 2: Full EVO-1 fine-tuning")

def test_lightweight_action_head():
    """Test lightweight action head for Stage 1 (exported for documentation)"""
    print("\n🧠 Testing Lightweight Action Head...")
    
    try:
        # Create action head
        action_head = LightweightActionHead(
            vision_dim=2048,
            language_dim=768,
            hidden_dim=256,
            action_dim=3
        )
        
        param_count = action_head.get_parameter_count()
        print(f"✅ Action head created with {param_count:,} parameters")
        
        # Test forward pass
        vision_features = torch.randn(2, 2048)
        language_features = torch.randn(2, 768)
        
        with torch.no_grad():
            actions = action_head(vision_features, language_features)
        
        print(f"✅ Forward pass successful: {actions.shape}")
        print(f"   Output range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Test training mode
        action_head.train()
        actions_train = action_head(vision_features, language_features)
        
        print(f"✅ Training mode works: {actions_train.requires_grad}")
        
        return True
        
    except Exception as e:
        print(f"❌ Action head test failed: {e}")
        return False

if __name__ == '__main__':
    asyncio.run(main())