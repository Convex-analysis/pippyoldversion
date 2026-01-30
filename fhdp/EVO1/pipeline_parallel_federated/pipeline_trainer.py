"""
Pipeline Parallel Training with FHDP Integration

This module orchestrates EVO-1 pipeline training using FHDP's
native system architecture and coordination capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from core.fhdp_system import FHDPSystem, SystemConfiguration
from core.types import VehicleInfo, ModelUpdate
from edge_server.server import EdgeServer
from vehicle_layer.vehicle import Vehicle
import logging
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass
from pathlib import Path

# Import EVO-1 components
try:
    from model.evo1_driving import EVO1Driving, ModelConfig
except ImportError:
    logging.warning("EVO-1 model not found. Using fallback.")

# Import pipeline components
from .edge_server.edge_integration import EdgeServerVLMIntegration
from .vehicle_client.vehicle_encoder import VehicleEncoderClient
from .coordinator.fhdp_coordinator import FHDPipelineCoordinator
from .adapter.hardware_integration import HardwareResourceAdapter


@dataclass
class FHDPipelineConfig:
    """Configuration for EVO-1 pipeline training with FHDP"""
    experiment_name: str = "evo1_fhd_pipeline"
    output_dir: str = "./outputs/evo1_fhd_pipeline"
    
    # FHDP System Configuration
    max_vehicles: int = 10
    max_vehicles_per_region: int = 20
    pipeline_formation_interval: float = 5.0
    aggregation_interval: float = 15.0
    fairness_enabled: bool = True
    
    # EVO-1 Training Configuration
    encoder_learning_rate: float = 1e-3
    action_head_learning_rate: float = 1e-3
    batch_size: int = 8
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Pipeline Configuration
    vlm_backbone_frozen: bool = True
    encoder_training: bool = True
    action_head_training: bool = True


class FHDPipelineTrainer:
    """
    Pipeline parallel trainer for EVO-1 using FHDP's native architecture
    
    This class orchestrates the entire training process using FHDP's
    existing system components while providing EVO-1 specific functionality.
    """
    
    def __init__(self, config: FHDPipelineConfig):
        self.config = config
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FHDP system
        self.fhdp_system = FHDPSystem(SystemConfiguration(
            max_vehicles_per_region=config.max_vehicles_per_region,
            pipeline_formation_interval=config.pipeline_formation_interval,
            aggregation_interval=config.aggregation_interval,
            fairness_enabled=config.fairness_enabled
        ))
        
        # Initialize pipeline components
        self.edge_servers = {}
        self.vehicle_clients = {}
        self.coordinator = FHDPipelineCoordinator()
        
        # Training state
        self.current_round = 0
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        logging.info(f"FHDPipeline Trainer initialized: {config.experiment_name}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.experiment_dir / "pipeline_training.log"
        
        # Configure logger
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.config.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    async def initialize_system(self, edge_server_configs: List[Dict], 
                             vehicle_configs: List[Dict]) -> bool:
        """Initialize the complete FHDP pipeline system"""
        try:
            self.logger.info("Initializing FHDP pipeline system...")
            
            # Initialize edge servers
            await self._initialize_edge_servers(edge_server_configs)
            
            # Initialize vehicle clients
            await self._initialize_vehicle_clients(vehicle_configs)
            
            # Start FHDP system
            self.fhdp_system.start_system()
            
            # Register vehicles with FHDP
            await self._register_vehicles_with_fhdp()
            
            self.logger.info("FHDP pipeline system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def _initialize_edge_servers(self, server_configs: List[Dict]):
        """Initialize edge servers with VLM integration"""
        for server_config in server_configs:
            server_id = server_config['server_id']
            
            # Create edge server with VLM integration
            edge_server = EdgeServerVLMIntegration(
                server_id=server_id,
                vlm_config=server_config.get('vlm_config')
            )
            
            # Start the server
            edge_server.start_server()
            
            self.edge_servers[server_id] = edge_server
            self.logger.info(f"Edge server initialized: {server_id}")
    
    async def _initialize_vehicle_clients(self, vehicle_configs: List[Dict]):
        """Initialize vehicle encoder clients"""
        for vehicle_config in vehicle_configs:
            vehicle_id = vehicle_config['vehicle_id']
            
            # Find appropriate edge server
            edge_server_id = vehicle_config.get('edge_server_id', 'edge_server_0')
            edge_server_url = f"http://localhost:808{edge_server_id.split('_')[-1]}"
            
            # Create vehicle client
            vehicle_client = VehicleEncoderClient(
                vehicle_id=vehicle_id,
                edge_server_url=edge_server_url,
                encoder_config=vehicle_config.get('encoder_config')
            )
            
            self.vehicle_clients[vehicle_id] = vehicle_client
            self.logger.info(f"Vehicle client initialized: {vehicle_id}")
    
    async def _register_vehicles_with_fhdp(self):
        """Register all vehicles with FHDP system"""
        for vehicle_id, vehicle_client in self.vehicle_clients.items():
            vehicle_info = {
                'vehicle_id': vehicle_id,
                'position': (0.0, 0.0),  # Default position
                'resources': vehicle_client._get_vehicle_resources(),
                'encoder_type': vehicle_client.training_state.get('encoder_type', 'resnet18')
            }
            
            success = self.coordinator.register_vehicle(vehicle_id, vehicle_info)
            if success:
                self.logger.info(f"Vehicle registered with FHDP: {vehicle_id}")
            else:
                self.logger.error(f"Failed to register vehicle: {vehicle_id}")
    
    async def start_pipeline_training(self, num_rounds: int) -> bool:
        """Start pipeline parallel training"""
        try:
            self.logger.info(f"Starting pipeline parallel training for {num_rounds} rounds")
            
            # Get list of vehicle IDs
            vehicle_ids = list(self.vehicle_clients.keys())
            
            # Start training rounds
            for round_num in range(num_rounds):
                success = await self._execute_training_round(round_num, vehicle_ids)
                
                if not success:
                    self.logger.error(f"Training round {round_num} failed")
                    continue
                
                # Save checkpoint periodically
                if round_num % 5 == 0:
                    await self._save_checkpoint(round_num)
            
            self.logger.info("Pipeline parallel training completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline training failed: {e}")
            return False
    
    async def _execute_training_round(self, round_num: int, vehicle_ids: List[str]) -> bool:
        """Execute a single training round"""
        self.logger.info(f"Executing training round {round_num + 1}")
        round_start_time = time.time()
        
        try:
            # Step 1: Start pipeline training with FHDP
            pipeline_config = {
                'encoder_training': self.config.encoder_training,
                'vlm_backbone_frozen': self.config.vlm_backbone_frozen,
                'action_head_training': self.config.action_head_training
            }
            
            pipeline_success = self.coordinator.start_pipeline_training(
                vehicle_ids, pipeline_config
            )
            
            if not pipeline_success:
                self.logger.error(f"Failed to start pipeline for round {round_num}")
                return False
            
            # Step 2: Train vehicles locally (simulated)
            vehicle_updates = {}
            for vehicle_id in vehicle_ids:
                vehicle_client = self.vehicle_clients[vehicle_id]
                
                # Simulate training epoch
                training_metrics = {
                    'loss': 0.5 - round_num * 0.01,  # Mock decreasing loss
                    'accuracy': 0.6 + round_num * 0.008,  # Mock increasing accuracy
                    'vehicle_id': vehicle_id,
                    'round': round_num
                }
                
                # Generate model update
                model_update = vehicle_client.generate_model_update()
                vehicle_updates[vehicle_id] = model_update
                
                self.logger.info(f"Vehicle {vehicle_id} training completed: loss={training_metrics['loss']:.3f}")
            
            # Step 3: Aggregate updates using FHDP
            updates_list = list(vehicle_updates.values())
            aggregated_update = self.coordinator.aggregate_updates(updates_list)
            
            if aggregated_update:
                # Step 4: Broadcast aggregated model
                pipeline_id = f"evo1_pipeline_{round_num}"
                broadcast_success = self.coordinator.broadcast_aggregated_model(
                    aggregated_update, pipeline_id
                )
                
                if broadcast_success:
                    # Step 5: Apply aggregated model to vehicles
                    for vehicle_id, vehicle_client in self.vehicle_clients.items():
                        vehicle_client.apply_aggregated_model(aggregated_update)
            
            # Record round metrics
            round_time = time.time() - round_start_time
            self._record_round_metrics(round_num, vehicle_ids, round_time, training_metrics)
            
            self.logger.info(f"Round {round_num + 1} completed in {round_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in training round {round_num}: {e}")
            return False
    
    def _record_round_metrics(self, round_num: int, vehicle_ids: List[str], 
                           round_time: float, training_metrics: Dict[str, Any]):
        """Record metrics for the training round"""
        round_metrics = {
            'round': round_num,
            'participating_vehicles': vehicle_ids,
            'num_vehicles': len(vehicle_ids),
            'round_time': round_time,
            'average_loss': training_metrics['loss'],
            'average_accuracy': training_metrics['accuracy'],
            'timestamp': time.time()
        }
        
        self.training_history.append(round_metrics)
        
        # Log summary
        self.logger.info(
            f"Round {round_num + 1} Summary: "
            f"Vehicles={len(vehicle_ids)}, "
            f"Avg Loss={training_metrics['loss']:.3f}, "
            f"Avg Accuracy={training_metrics['accuracy']:.3f}, "
            f"Time={round_time:.2f}s"
        )
    
    async def _save_checkpoint(self, round_num: int):
        """Save training checkpoint"""
        try:
            checkpoint_path = self.experiment_dir / f"checkpoint_round_{round_num}.pt"
            
            # Prepare checkpoint data
            checkpoint = {
                'round': round_num,
                'config': self.config,
                'training_history': self.training_history,
                'coordinator_status': self.coordinator.get_coordinator_status(),
                'vehicle_states': {
                    vid: client.get_vehicle_status() 
                    for vid, client in self.vehicle_clients.items()
                },
                'timestamp': time.time()
            }
            
            # Save checkpoint
            import torch
            torch.save(checkpoint, checkpoint_path)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        if not self.training_history:
            return {}
        
        # Calculate overall statistics
        total_rounds = len(self.training_history)
        total_time = sum(r['round_time'] for r in self.training_history)
        avg_round_time = total_time / total_rounds if total_rounds > 0 else 0
        
        final_metrics = self.training_history[-1] if self.training_history else {}
        
        return {
            'experiment_name': self.config.experiment_name,
            'total_rounds': total_rounds,
            'total_training_time': total_time,
            'average_round_time': avg_round_time,
            'final_loss': final_metrics.get('average_loss', 0),
            'final_accuracy': final_metrics.get('average_accuracy', 0),
            'vehicles_participated': len(self.vehicle_clients),
            'edge_servers_used': len(self.edge_servers),
            'coordinator_status': self.coordinator.get_coordinator_status(),
            'training_history': self.training_history[-10:]  # Last 10 rounds
        }
    
    def save_final_results(self):
        """Save final training results"""
        try:
            results_path = self.experiment_dir / "final_results.json"
            
            results = {
                'config': self.config.__dict__,
                'statistics': self.get_training_statistics(),
                'coordinator_final_status': self.coordinator.get_coordinator_status(),
                'experiment_completed_at': time.time()
            }
            
            import json
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Final results saved: {results_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
    
    async def shutdown_system(self):
        """Shutdown the complete FHDP pipeline system"""
        try:
            self.logger.info("Shutting down FHDP pipeline system...")
            
            # Shutdown vehicle clients
            for vehicle_client in self.vehicle_clients.values():
                vehicle_client.cleanup()
            
            # Shutdown edge servers
            for edge_server in self.edge_servers.values():
                edge_server.stop_server()
            
            # Shutdown coordinator
            self.coordinator.shutdown_coordinator()
            
            # Shutdown FHDP system
            self.fhdp_system.shutdown_system()
            
            # Save final results
            self.save_final_results()
            
            self.logger.info("FHDP pipeline system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Convenience function for running FHDP pipeline training
async def run_fhd_pipeline_training(config: FHDPipelineConfig,
                                 edge_configs: List[Dict],
                                 vehicle_configs: List[Dict],
                                 num_rounds: int = 50) -> Dict[str, Any]:
    """Convenience function to run FHDP pipeline training"""
    trainer = FHDPipelineTrainer(config)
    
    try:
        # Initialize system
        init_success = await trainer.initialize_system(edge_configs, vehicle_configs)
        if not init_success:
            raise Exception("Failed to initialize FHDP pipeline system")
        
        # Start training
        training_success = await trainer.start_pipeline_training(num_rounds)
        if not training_success:
            raise Exception("Pipeline training failed")
        
        # Get statistics
        return trainer.get_training_statistics()
        
    except Exception as e:
        logging.error(f"FHDP pipeline training failed: {e}")
        raise
    finally:
        await trainer.shutdown_system()