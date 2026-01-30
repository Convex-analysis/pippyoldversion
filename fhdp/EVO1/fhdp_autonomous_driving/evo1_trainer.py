"""
EVO-1 FHDP Autonomous Driving Trainer

Core trainer that integrates EVO-1 model with FHDP system for
distributed autonomous driving training with real-time coordination.
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import FHDP core components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
try:
    from core.fhdp_system import FHDPSystem, SystemConfiguration
    from core.types import VehicleInfo, TrainingMode, ModelUpdate, AggregationResult
    from edge_server.server import EdgeServer
    from vehicle_layer.vehicle import Vehicle
    FHDP_AVAILABLE = True
except ImportError:
    logging.warning("FHDP core components not available. Using standalone mode.")
    FHDP_AVAILABLE = False
    SystemConfiguration = None

# Import EVO-1 components
from ..model.evo1_driving import FederatedEVO1Driving, EVO1DrivingOutput
from ..data.nuscenes_loader import create_dataloader
from ..data.augmentation import DrivingAugmentation, create_comprehensive_augmentation
from ..utils.config import EVO1DrivingConfig


@dataclass
class AutonomousDrivingState:
    """Training state for autonomous driving with FHDP"""
    current_round: int = 0
    total_rounds: int = 100
    active_vehicles: List[str] = None
    coordinated_vehicles: List[str] = None
    pipeline_formations: Dict[str, List[str]] = None
    global_metrics: Dict[str, float] = None
    autonomous_metrics: Dict[str, float] = None  # Driving-specific metrics
    
    def __post_init__(self):
        if self.active_vehicles is None:
            self.active_vehicles = []
        if self.coordinated_vehicles is None:
            self.coordinated_vehicles = []
        if self.pipeline_formations is None:
            self.pipeline_formations = {}
        if self.global_metrics is None:
            self.global_metrics = {}
        if self.autonomous_metrics is None:
            self.autonomous_metrics = {}


class FHDAutonomousDrivingTrainer:
    """EVO-1 trainer with FHDP coordination for autonomous driving"""
    
    def __init__(
        self,
        config: EVO1DrivingConfig,
        fhdp_config: Optional[SystemConfiguration] = None,
        device: str = "cuda"
    ):
        self.config = config
        self.device = device
        self.fhdp_config = fhdp_config
        
        print(f"[FHDP_AUTO] Initializing EVO-1 FHDP Autonomous Driving Trainer...")
        
        # Initialize FHDP system if available
        if FHDP_AVAILABLE and self.fhdp_config:
            print(f"[FHDP_AUTO] Setting up FHDP system for autonomous driving...")
            self.fhdp_system = FHDPSystem(self.fhdp_config)
            self.coordination_manager = self.fhdp_system.coordination_manager
            self.aggregation_manager = self.fhdp_system.aggregation_manager
            print(f"[FHDP_AUTO] FHDP system initialized successfully")
        else:
            print(f"[FHDP_AUTO] FHDP not available, using standalone autonomous mode")
            self.fhdp_system = None
            self.coordination_manager = None
            self.aggregation_manager = None
        
        # Initialize training state
        self.training_state = AutonomousDrivingState(
            current_round=0,
            total_rounds=config.training.aggregation_rounds
        )
        
        # Setup global EVO-1 model
        self.setup_global_model()
        
        # Setup autonomous vehicles
        self.setup_autonomous_vehicles()
        
        # Setup output directories
        self.setup_output_dirs()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb if enabled
        self.setup_wandb()
        
        # Load checkpoint if specified
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)
    
    def setup_global_model(self):
        """Setup global EVO-1 model for autonomous driving"""
        
        self.global_model = FederatedEVO1Driving(
            config=self.config.model,
            training_config=self.config.training,
            device=self.device
        ).to(self.device)
        
        print(f"[FHDP_AUTO] Global EVO-1 model setup complete")
        print(f"[FHDP_AUTO] Model parameters: {sum(p.numel() for p in self.global_model.parameters()):,}")
    
    def setup_autonomous_vehicles(self):
        """Setup autonomous driving vehicles with EVO-1 models"""
        
        num_vehicles = self.config.training.num_clients
        print(f"[FHDP_AUTO] Setting up {num_vehicles} autonomous vehicles...")
        
        self.vehicles = {}
        self.vehicle_trainers = {}
        
        for i in range(num_vehicles):
            vehicle_id = f"auto_vehicle_{i}"
            
            # Create vehicle info for autonomous driving
            vehicle_info = VehicleInfo(
                vehicle_id=vehicle_id,
                vehicle_type="autonomous_driving_evo1",
                model_type="EVO1",
                capabilities=[
                    "vision_perception",
                    "action_prediction", 
                    "path_planning",
                    "vehicle_control",
                    "scene_understanding",
                    "multi_modal_reasoning"
                ],
                resource_class="high",  # Autonomous driving requires high resources
                location=f"deployment_region_{i % 4}",  # Distribute across 4 regions
                status="active"
            )
            
            # Create vehicle with EVO-1 model
            vehicle = self.create_autonomous_vehicle(vehicle_info)
            
            # Store vehicle
            self.vehicles[vehicle_id] = vehicle
            self.vehicle_trainers[vehicle_id] = vehicle
        
        # Register vehicles with FHDP system
        if self.fhdp_system:
            self.fhdp_system.register_vehicles(list(self.vehicles.values()))
            print(f"[FHDP_AUTO] Registered {len(self.vehicles)} vehicles with FHDP system")
        
        print(f"[FHDP_AUTO] Autonomous vehicles setup complete")
    
    def create_autonomous_vehicle(self, vehicle_info):
        """Create an autonomous driving vehicle with EVO-1"""
        
        if self.fhdp_system:
            # Create FHDP-managed vehicle
            from vehicle_layer.vehicle import Vehicle
            return Vehicle(
                vehicle_info=vehicle_info,
                model=self.global_model,
                device=self.device
            )
        else:
            # Create standalone vehicle
            return self.create_standalone_vehicle(vehicle_info)
    
    def create_standalone_vehicle(self, vehicle_info):
        """Create standalone autonomous driving vehicle"""
        
        # Create local optimizer
        optimizer = optim.AdamW(
            self.global_model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Create data loader for this vehicle
        try:
            vehicle_num = int(vehicle_info.vehicle_id.split('_')[-1])
        except (IndexError, ValueError):
            vehicle_num = 0
        
        train_loader = create_dataloader(
            config=self.config.data,
            model_config=self.config.model,
            split="train",
            client_id=vehicle_num,
            num_clients=self.config.training.num_clients,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = create_dataloader(
            config=self.config.data,
            model_config=self.config.model,
            split="val",
            client_id=vehicle_num,
            num_clients=self.config.training.num_clients,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Create vehicle wrapper for standalone mode
        return StandaloneAutonomousVehicle(
            vehicle_info=vehicle_info,
            model=self.global_model.clone(),
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            config=self.config
        )
    
    def setup_output_dirs(self):
        """Setup output directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "fhdp"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "autonomous"), exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.output_dir, "logs", "fhdp_autonomous_driving.log")
        
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if os.getenv("WANDB_API_KEY"):
            import wandb
            wandb.init(
                project="evo1-fhdp-autonomous-driving",
                name=f"{self.config.experiment_name}_autonomous",
                config={
                    **self.config.__dict__,
                    "training_mode": "fhdp_autonomous_driving",
                    "fhdp_enabled": self.fhdp_system is not None,
                    "num_vehicles": len(self.vehicles)
                }
            )
            logging.info("Initialized wandb logging")
        else:
            logging.info("Wandb not configured. Skipping wandb logging.")
    
    def autonomous_federated_round(self, round_idx: int) -> Dict[str, float]:
        """Execute one federated learning round with autonomous vehicles"""
        
        logging.info(f"Starting FHDP autonomous driving round {round_idx + 1}/{self.training_state.total_rounds}")
        
        round_start_time = time.time()
        round_metrics = {}
        
        # Use FHDP system to select and coordinate vehicles
        if self.fhdp_system:
            # FHDP manages vehicle selection and coordination
            participating_vehicles = self.fhdp_system.select_participating_vehicles(
                round_idx=round_idx,
                client_fraction=self.config.training.client_fraction
            )
            
            self.training_state.active_vehicles = [v.vehicle_id for v in participating_vehicles]
            
            # Get vehicles selected for pipeline coordination
            coordinated_vehicles = self.fhdp_system.get_pipeline_participants(round_idx)
            self.training_state.coordinated_vehicles = [v.vehicle_id for v in coordinated_vehicles]
            
            # Get pipeline formations
            pipeline_formations = self.fhdp_system.get_pipeline_formations(round_idx)
            self.training_state.pipeline_formations = {
                formation_id: [v.vehicle_id for v in vehicles]
                for formation_id, vehicles in pipeline_formations.items()
            }
            
            logging.info(f"FHDP selected {len(participating_vehicles)} vehicles for individual training")
            logging.info(f"FHDP selected {len(coordinated_vehicles)} vehicles for pipeline coordination")
            
        else:
            # Standalone mode: use all vehicles
            self.training_state.active_vehicles = list(self.vehicles.keys())
            self.training_state.coordinated_vehicles = []
            self.training_state.pipeline_formations = {}
        
        # Local training on active vehicles
        vehicle_updates = {}
        vehicle_metrics = {}
        
        for vehicle_id in self.training_state.active_vehicles:
            vehicle = self.vehicles[vehicle_id]
            
            logging.info(f"Training autonomous vehicle {vehicle_id}")
            
            # Local training
            metrics = self.train_vehicle_locally(vehicle, round_idx)
            
            # Get model updates
            updates = self.get_vehicle_updates(vehicle)
            
            vehicle_updates[vehicle_id] = updates
            vehicle_metrics[vehicle_id] = metrics
        
        # Use FHDP system for aggregation
        if self.fhdp_system:
            # Convert to FHDP format
            fhdp_updates = {}
            for vehicle_id, updates in vehicle_updates.items():
                fhdp_updates[vehicle_id] = ModelUpdate(
                    vehicle_id=vehicle_id,
                    update_data=updates,
                    round_idx=round_idx,
                    timestamp=time.time(),
                    update_type="autonomous_driving_model_update"
                )
            
            # Perform FHDP aggregation
            aggregation_result = self.fhdp_system.aggregate_model_updates(fhdp_updates)
            
            # Apply aggregated updates to global model
            if aggregation_result.aggregated_update:
                self.global_model.update_from_aggregation(aggregation_result.aggregated_update)
            
            # Update vehicle models
            for vehicle_id in self.training_state.active_vehicles:
                self.vehicles[vehicle_id].model.set_global_model(self.global_model.state_dict())
            
            # Update training state with FHDP metrics
            self.training_state.global_metrics.update({
                'fhdp_participants': len(self.training_state.active_vehicles),
                'fhdp_coordinated': len(self.training_state.coordinated_vehicles),
                'fhdp_pipeline_formations': len(self.training_state.pipeline_formations),
                'fhdp_aggregation_time': getattr(aggregation_result, 'aggregation_time', 0.0),
                'fhdp_fairness_score': getattr(aggregation_result, 'fairness_score', 0.0)
            })
            
            logging.info(f"FHDP aggregation completed in {aggregation_result.aggregation_time:.2f}s")
        else:
            # Standalone aggregation (FedAvg)
            aggregated_updates = self.standalone_aggregation(vehicle_updates)
            self.global_model.update_from_aggregation(aggregated_updates)
            
            # Update vehicle models
            for vehicle_id in self.training_state.active_vehicles:
                self.vehicles[vehicle_id].model.set_global_model(self.global_model.state_dict())
            
            self.training_state.global_metrics.update({
                'fhdp_participants': len(self.training_state.active_vehicles),
                'fhdp_aggregation_time': 0.0
            })
        
        # Compute autonomous driving specific metrics
        self.compute_autonomous_metrics(vehicle_metrics, round_idx)
        
        # Compute round metrics
        round_time = time.time() - round_start_time
        round_metrics.update(self.training_state.global_metrics)
        round_metrics['round_time'] = round_time
        round_metrics['autonomous_driving_score'] = self.training_state.autonomous_metrics.get('driving_score', 0.0)
        round_metrics['coordination_efficiency'] = self.training_state.autonomous_metrics.get('coordination_efficiency', 0.0)
        
        # Update training state
        self.training_state.current_round = round_idx
        
        # Log metrics
        self.log_round_metrics(round_idx, round_metrics, vehicle_metrics)
        
        # Save checkpoint
        if (round_idx + 1) % self.config.save_frequency == 0:
            self.save_autonomous_checkpoint(round_idx)
        
        return round_metrics
    
    def train_vehicle_locally(self, vehicle, round_idx: int) -> Dict[str, float]:
        """Train vehicle locally for autonomous driving"""
        
        # Handle both FHDP-managed and standalone vehicles
        if hasattr(vehicle, 'train_locally'):
            return vehicle.train_locally(round_idx)
        
        # Standalone vehicle training
        vehicle.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(vehicle.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Apply augmentation
            if self.config.training.federated_learning:
                batch = vehicle.augmentation.augment_batch(batch) if hasattr(vehicle, 'augmentation') else batch
            
            # Forward pass
            vehicle.optimizer.zero_grad()
            
            if self.config.training.mixed_precision and hasattr(vehicle, 'scaler'):
                with autocast():
                    output = vehicle.model(
                        images=batch['images'],
                        image_mask=batch['image_mask'],
                        state=batch['state'],
                        instructions=batch['instructions'],
                        future_controls=batch['future_controls'],
                        mode="training"
                    )
                    
                    loss_dict = vehicle.model.compute_loss(
                        output=output,
                        target_controls=batch['future_controls']
                    )
                    loss = loss_dict['total_loss']
                
                # Backward pass
                vehicle.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    vehicle.scaler.unscale_(vehicle.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in vehicle.model.parameters() if p.requires_grad], 
                        self.config.training.max_grad_norm
                    )
                
                vehicle.scaler.step(vehicle.optimizer)
                vehicle.scaler.update()
            else:
                # Standard precision training
                output = vehicle.model(
                    images=batch['images'],
                    image_mask=batch['image_mask'],
                    state=batch['state'],
                    instructions=batch['instructions'],
                    future_controls=batch['future_controls'],
                    mode="training"
                )
                
                loss_dict = vehicle.model.compute_loss(
                    output=output,
                    target_controls=batch['future_controls']
                )
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in vehicle.model.parameters() if p.requires_grad], 
                        self.config.training.max_grad_norm
                    )
                
                vehicle.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                logging.info(
                    f"Vehicle {vehicle.vehicle_info.vehicle_id} - Round {round_idx} - "
                    f"Batch {batch_idx}/{len(vehicle.train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Compute epoch metrics
        epoch_metrics = {
            'train_loss': total_loss / max(num_batches, 1),
            'learning_rate': vehicle.optimizer.param_groups[0]['lr']
        }
        
        return epoch_metrics
    
    def get_vehicle_updates(self, vehicle) -> Dict[str, torch.Tensor]:
        """Get model updates from vehicle"""
        
        if hasattr(vehicle, 'get_local_updates'):
            return vehicle.get_local_updates()
        elif hasattr(vehicle, 'model'):
            return vehicle.model.get_local_updates()
        else:
            # Standalone vehicle: return all parameters
            return vehicle.model.state_dict()
    
    def compute_autonomous_metrics(self, vehicle_metrics: Dict[str, Dict[str, float]], round_idx: int):
        """Compute autonomous driving specific metrics"""
        
        if not vehicle_metrics:
            self.training_state.autonomous_metrics = {}
            return
        
        # Compute driving-related metrics
        avg_train_loss = np.mean([
            metrics.get('train_loss', 0) 
            for metrics in vehicle_metrics.values()
        ])
        
        # Simulate driving score (loss-based, for now)
        driving_score = max(0.0, 1.0 - avg_train_loss / 0.5)  # Higher score = better driving
        
        # Coordination efficiency (for FHDP systems)
        coordination_efficiency = 1.0  # Will be updated by FHDP system
        
        self.training_state.autonomous_metrics = {
            'driving_score': driving_score,
            'coordination_efficiency': coordination_efficiency,
            'vehicle_performance_score': np.mean([
                metrics.get('train_loss', 0) 
                for metrics in vehicle_metrics.values()
            ]),
            'fleet_coordination': len(self.training_state.coordinated_vehicles) / max(len(self.training_state.active_vehicles), 1)
        }
    
    def standalone_aggregation(self, vehicle_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Standalone FedAvg aggregation"""
        
        if not vehicle_updates:
            return {}
        
        # Simple FedAvg - equal weights for all vehicles
        vehicle_weights = [1.0 / len(vehicle_updates)] * len(vehicle_updates)
        
        # Use first vehicle's update structure as template
        aggregated_updates = {}
        
        first_vehicle = list(vehicle_updates.keys())[0]
        for param_name in vehicle_updates[first_vehicle]:
            # Weighted average of parameter updates
            param_updates = [vehicle_updates[vehicle][param_name] for vehicle in vehicle_updates]
            aggregated = sum(w * u for w, u in zip(vehicle_weights, param_updates))
            aggregated_updates[param_name] = aggregated
        
        logging.info(f"[STANDALONE_AGGR] Aggregated {len(aggregated_updates)} parameters")
        return aggregated_updates
    
    def log_round_metrics(self, round_idx: int, global_metrics: Dict[str, float], 
                         vehicle_metrics: Dict[str, Dict[str, float]]):
        """Log metrics for current round"""
        
        # Log to console
        logging.info(f"FHDP Autonomous Driving Round {round_idx + 1} completed:")
        for metric, value in global_metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
        
        # Log to wandb
        if os.getenv("WANDB_API_KEY"):
            import wandb
            wandb.log({
                "round": round_idx,
                **{f"global/{k}": v for k, v in global_metrics.items()}
            })
            
            # Log vehicle metrics
            for vehicle_id, metrics in vehicle_metrics.items():
                wandb.log({
                    f"vehicle_{vehicle_id}/train_loss": metrics.get('train_loss', 0),
                    f"vehicle_{vehicle_id}/driving_score": self.training_state.autonomous_metrics.get('driving_score', 0),
                    "round": round_idx
                })
        
        # Save metrics to file
        metrics_file = os.path.join(
            self.config.output_dir, "fhdp", 
            f"autonomous_round_{round_idx:04d}_metrics.json"
        )
        
        metrics_data = {
            'round': round_idx,
            'global_metrics': global_metrics,
            'vehicle_metrics': {k: {str(mk): float(mv) for mk, mv in v.items()} 
                              for k, v in vehicle_metrics.items()},
            'autonomous_metrics': self.training_state.autonomous_metrics,
            'fhdp_system_state': {
                'active_vehicles': self.training_state.active_vehicles,
                'coordinated_vehicles': self.training_state.coordinated_vehicles,
                'pipeline_formations': self.training_state.pipeline_formations
            },
            'timestamp': time.time()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def save_autonomous_checkpoint(self, round_idx: int):
        """Save autonomous driving checkpoint"""
        
        checkpoint_path = os.path.join(
            self.config.output_dir, "checkpoints",
            f"autonomous_round_{round_idx:04d}.pt"
        )
        
        self.global_model.save_checkpoint(
            filepath=checkpoint_path,
            epoch=round_idx,
            optimizer_state=None
        )
        
        logging.info(f"Saved autonomous driving checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load global model state
            self.global_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set training state
            self.training_state.current_round = checkpoint.get('epoch', 0) + 1
            
            logging.info(f"Loaded autonomous driving checkpoint from round {self.training_state.current_round}")
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
    
    def train(self):
        """Main training loop for autonomous driving"""
        
        print("[FHDP_AUTO] Starting FHDP Autonomous Driving Training...")
        if self.fhdp_system:
            print("[FHDP_AUTO] Using FHDP system for coordinated autonomous driving")
        else:
            print("[FHDP_AUTO] Using standalone autonomous driving mode")
        
        logging.info("Starting FHDP autonomous driving training")
        
        start_time = time.time()
        logging.info(f"Starting autonomous driving training for {self.config.training.aggregation_rounds} rounds with {len(self.vehicles)} vehicles")
        
        try:
            # Start FHDP system if available
            if self.fhdp_system:
                self.fhdp_system.start()
                logging.info("FHDP system started for autonomous driving coordination")
            
            for round_idx in range(self.training_state.current_round, self.config.training.aggregation_rounds):
                logging.info(f"Starting autonomous driving round {round_idx + 1}/{self.config.training.aggregation_rounds}")
                
                round_metrics = self.autonomous_federated_round(round_idx)
                logging.info(f"Round {round_idx + 1} completed: avg_loss={round_metrics.get('avg_train_loss', 0):.4f}")
                
                # Early stopping if loss is low enough
                if round_metrics.get('avg_train_loss', float('inf')) < 0.05:
                    logging.info("Early stopping: driving loss below threshold")
                    break
            
            # Stop FHDP system
            if self.fhdp_system:
                self.fhdp_system.stop()
                logging.info("FHDP system stopped")
            
            # Save final model
            self.save_autonomous_checkpoint(self.config.training.aggregation_rounds - 1)
            
            total_time = time.time() - start_time
            logging.info(f"FHDP autonomous driving training completed in {total_time:.2f} seconds")
            
            if os.getenv("WANDB_API_KEY"):
                import wandb
                wandb.log({"total_training_time": total_time})
                wandb.finish()
            
            print("\n[FHDP_AUTO] Autonomous driving training completed successfully!")
            
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_autonomous_checkpoint(self.training_state.current_round)
            
            if self.fhdp_system:
                self.fhdp_system.stop()
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            
            if self.fhdp_system:
                self.fhdp_system.stop()
            
            raise


class StandaloneAutonomousVehicle:
    """Standalone autonomous driving vehicle wrapper"""
    
    def __init__(
        self,
        vehicle_info,
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        config
    ):
        self.vehicle_info = vehicle_info
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Create augmentation
        self.augmentation = create_comprehensive_augmentation()
        
        # Create scaler for mixed precision
        self.scaler = GradScaler() if config.training.mixed_precision else None
    
    def train_locally(self, round_idx: int) -> Dict[str, float]:
        """Train vehicle locally"""
        
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Apply augmentation
            if self.config.training.federated_learning:
                batch = self.augmentation.augment_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.training.mixed_precision and self.scaler is not None:
                with autocast():
                    output = self.model(
                        images=batch['images'],
                        image_mask=batch['image_mask'],
                        state=batch['state'],
                        instructions=batch['instructions'],
                        future_controls=batch['future_controls'],
                        mode="training"
                    )
                    
                    loss_dict = self.model.compute_loss(
                        output=output,
                        target_controls=batch['future_controls']
                    )
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        self.config.training.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                output = self.model(
                    images=batch['images'],
                    image_mask=batch['image_mask'],
                    state=batch['state'],
                    instructions=batch['instructions'],
                    future_controls=batch['future_controls'],
                    mode="training"
                )
                
                loss_dict = self.model.compute_loss(
                    output=output,
                    target_controls=batch['future_controls']
                )
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                logging.info(
                    f"Standalone Vehicle {self.vehicle_info.vehicle_id} - Round {round_idx} - "
                    f"Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Compute epoch metrics
        epoch_metrics = {
            'train_loss': total_loss / max(num_batches, 1),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return epoch_metrics