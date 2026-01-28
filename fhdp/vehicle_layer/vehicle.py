"""
Vehicle Implementation for FHDP System

Implements complete vehicle functionality with all FHDP components.
"""
import time
import threading
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .communication import V2VCommunicationManager
from .pipeline_formation import PipelineFormation
from .training_engine import TrainingExecutor
from .monitor import VehicleMonitor
from core.types import (
    VehicleInfo, VehicleState, TrainingMode, Pipeline, PipelineTemplate,
    ModelUpdate, ResourceMetrics, CommunicationBundle
)
from core.constants import HEARTBEAT_INTERVAL

class Vehicle:
    """Main Vehicle implementation with complete FHDP functionality"""
    
    def __init__(self, vehicle_id: str, initial_position: Tuple[float, float] = (0, 0),
                 initial_velocity: float = 0.0, initial_direction: float = 0.0,
                 resources: Optional[Dict[str, Any]] = None):
        # Vehicle information
        self.vehicle_info = VehicleInfo(
            vehicle_id=vehicle_id,
            position=initial_position,
            velocity=initial_velocity,
            direction=initial_direction,
            resources=resources or self._default_resources(),
            state=VehicleState.IDLE
        )
        
        # Initialize components
        self.communication_manager = V2VCommunicationManager(self.vehicle_info)
        self.pipeline_formation = PipelineFormation(self.vehicle_info)
        self.training_executor = TrainingExecutor(self.vehicle_info)
        self.vehicle_monitor = VehicleMonitor(self.vehicle_info)
        
        # Vehicle state
        self.current_state = VehicleState.IDLE
        self.active_pipeline: Optional[str] = None
        self.training_mode: Optional[TrainingMode] = None
        self.neighbors: Dict[str, Any] = {}
        
        # Callback handlers
        self.message_handlers = {
            'pipeline_invitation': self._handle_pipeline_invitation,
            'model_broadcast': self._handle_model_broadcast,
            'training_request': self._handle_training_request,
            'error_propagation': self._handle_error_propagation
        }
        
        # Statistics
        self.vehicle_stats = {
            'total_training_sessions': 0,
            'successful_sessions': 0,
            'pipeline_participations': 0,
            'individual_participations': 0,
            'neighbors_discovered': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'uptime': 0.0
        }
        
        # Threading
        self.heartbeat_thread = None
        self.vehicle_active = False
        self.stop_event = threading.Event()
    
    def _default_resources(self) -> Dict[str, Any]:
        """Default vehicle resource configuration"""
        return {
            'cpu': 0.7,           # CPU availability (0.0-1.0)
            'memory': 0.6,         # Memory availability
            'battery': 0.8,         # Battery level
            'network_quality': 0.8, # Network quality
            'thermal_state': 0.3,    # Thermal state
            'cpu_capacity': 2.0,     # CPU capacity in GHz
            'memory_capacity': 8.0,  # Memory capacity in GB
            'battery_capacity': 50.0, # Battery capacity in Wh
            'bandwidth': 100.0       # Network bandwidth in Mbps
        }
    
    def start_vehicle(self, protocols: List[str] = None):
        """Start vehicle FHDP functionality"""
        if self.vehicle_active:
            return
        
        print(f"Starting vehicle {self.vehicle_info.vehicle_id}...")
        
        # Initialize communication
        if protocols is None:
            protocols = ['dsrc']
        
        protocol_objects = []
        for protocol in protocols:
            from core.types import CommunicationProtocol
            try:
                protocol_objects.append(CommunicationProtocol(protocol))
            except ValueError:
                print(f"Unknown protocol: {protocol}")
        
        if protocol_objects:
            self.communication_manager.initialize(protocol_objects)
        
        # Start training executor
        self.training_executor.start_execution_service()
        
        # Start monitoring
        self.vehicle_monitor.start_monitoring()
        
        # Set up communication callbacks
        self._setup_communication_callbacks()
        
        # Start heartbeat
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        
        self.vehicle_active = True
        self.start_time = time.time()
        
        print(f"Vehicle {self.vehicle_info.vehicle_id} started successfully")
    
    def stop_vehicle(self):
        """Stop vehicle FHDP functionality"""
        if not self.vehicle_active:
            return
        
        print(f"Stopping vehicle {self.vehicle_info.vehicle_id}...")
        
        self.stop_event.set()
        
        # Stop communication
        self.communication_manager.shutdown()
        
        # Stop training executor
        self.training_executor.stop_execution_service()
        
        # Stop monitoring
        self.vehicle_monitor.stop_monitoring()
        
        # Stop heartbeat thread
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
        
        self.vehicle_active = False
        self.vehicle_stats['uptime'] = time.time() - self.start_time
        
        print(f"Vehicle {self.vehicle_info.vehicle_id} stopped")
    
    def _setup_communication_callbacks(self):
        """Set up communication event callbacks"""
        # Register message callbacks
        self.communication_manager.register_message_callback(
            'pipeline', self._handle_pipeline_message
        )
        self.communication_manager.register_message_callback(
            'model_update', self._handle_model_update_message
        )
        self.communication_manager.register_message_callback(
            'data', self._handle_data_message
        )
        
        # Register neighbor callback
        self.communication_manager.register_neighbor_callback(self._handle_neighbor_update)
    
    def _heartbeat_worker(self):
        """Send periodic heartbeat messages"""
        while not self.stop_event.is_set():
            try:
                self.communication_manager.broadcast_heartbeat()
                time.sleep(HEARTBEAT_INTERVAL)
            except Exception as e:
                print(f"Heartbeat error: {e}")
                time.sleep(HEARTBEAT_INTERVAL)
    
    def update_position(self, new_position: Tuple[float, float], 
                       new_velocity: float, new_direction: float):
        """Update vehicle position and mobility"""
        self.vehicle_info.position = new_position
        self.vehicle_info.velocity = new_velocity
        self.vehicle_info.direction = new_direction
        self.vehicle_info.last_seen = time.time()
    
    def update_resources(self, resources: Dict[str, Any]):
        """Update vehicle resources"""
        self.vehicle_info.resources.update(resources)
        
        # Update vehicle monitor resources
        resource_metrics = ResourceMetrics(
            cpu_usage=1.0 - resources.get('cpu', 0.7),
            memory_usage=1.0 - resources.get('memory', 0.6),
            battery_level=resources.get('battery', 0.8),
            network_quality=resources.get('network_quality', 0.8),
            thermal_state=resources.get('thermal_state', 0.3)
        )
        
        self.vehicle_monitor.record_training_participation(
            TrainingMode.INDIVIDUAL, "", None, 0.0, True, 1.0
        )  # Update participation tracking
    
    def _handle_neighbor_update(self, neighbor_id: str, neighbor_info):
        """Handle neighbor update events"""
        if neighbor_info:
            self.neighbors[neighbor_id] = neighbor_info
            if len(self.neighbors) > self.vehicle_stats['neighbors_discovered']:
                self.vehicle_stats['neighbors_discovered'] = len(self.neighbors)
        else:
            self.neighbors.pop(neighbor_id, None)
    
    def _handle_pipeline_message(self, message, sender_address):
        """Handle pipeline-related messages"""
        try:
            payload = message.payload
            
            if payload.get('type') == 'invitation':
                self._handle_pipeline_invitation(payload)
            elif payload.get('type') == 'formation_result':
                self._handle_pipeline_formation_result(payload)
            elif payload.get('type') == 'dissolution':
                self._handle_pipeline_dissolution(payload)
                
        except Exception as e:
            print(f"Pipeline message handling error: {e}")
    
    def _handle_model_update_message(self, message, sender_address):
        """Handle model update messages"""
        self.vehicle_stats['messages_received'] += 1
        
        try:
            # Process model update
            # In real implementation, this would update local model
            pass
            
        except Exception as e:
            print(f"Model update handling error: {e}")
    
    def _handle_data_message(self, message, sender_address):
        """Handle general data messages"""
        self.vehicle_stats['messages_received'] += 1
        # Generic message handling
    
    def _handle_pipeline_invitation(self, invitation_data: Dict[str, Any]):
        """Handle pipeline formation invitation"""
        try:
            pipeline_info = invitation_data.get('pipeline_info', {})
            template_data = pipeline_info.get('template', {})
            position = pipeline_info.get('position', -1)
            
            # Create template object
            from core.types import PipelineTemplate, TrainingConfig
            from core.types import ResourceClass
            
            template = PipelineTemplate(
                template_id=template_data.get('template_id', ''),
                resource_requirements=[
                    ResourceClass(r) for r in template_data.get('resource_requirements', [])
                ],
                expected_duration=template_data.get('expected_duration', 10.0),
                communication_pattern=template_data.get('communication_pattern', []),
                training_config=TrainingConfig(),
                model_fragment_size=template_data.get('model_fragment_size', 0)
            )
            
            # Accept or decline based on resources and state
            can_participate, reason = self.vehicle_monitor.can_participate_in_training()
            
            if can_participate and self.current_state == VehicleState.IDLE:
                # Accept invitation
                response = {
                    'type': 'invitation_response',
                    'vehicle_id': self.vehicle_info.vehicle_id,
                    'accepted': True,
                    'position': position
                }
                
                # Send response (in real implementation)
                self.current_state = VehicleState.PIPELINE
                self.active_pipeline = invitation_data.get('pipeline_id')
                
            else:
                # Decline invitation
                response = {
                    'type': 'invitation_response',
                    'vehicle_id': self.vehicle_info.vehicle_id,
                    'accepted': False,
                    'reason': reason
                }
            
            # Send response back to inviter (in real implementation)
            print(f"Pipeline invitation response: {response}")
            
        except Exception as e:
            print(f"Pipeline invitation handling error: {e}")
    
    def _handle_pipeline_formation_result(self, result_data: Dict[str, Any]):
        """Handle pipeline formation result"""
        pipeline_id = result_data.get('pipeline_id')
        success = result_data.get('success', False)
        
        if success:
            self.current_state = VehicleState.PIPELINE
            self.active_pipeline = pipeline_id
            self.training_mode = TrainingMode.PIPELINE
            print(f"Joined pipeline {pipeline_id}")
        else:
            self.current_state = VehicleState.IDLE
            print(f"Failed to join pipeline {pipeline_id}")
    
    def _handle_pipeline_dissolution(self, dissolution_data: Dict[str, Any]):
        """Handle pipeline dissolution"""
        pipeline_id = dissolution_data.get('pipeline_id')
        
        if self.active_pipeline == pipeline_id:
            self.current_state = VehicleState.IDLE
            self.active_pipeline = None
            self.training_mode = None
            print(f"Pipeline {pipeline_id} dissolved")
    
    def _handle_model_broadcast(self, model_data: Dict[str, Any]):
        """Handle global model broadcast"""
        # Update local model with global model
        # In real implementation, this would update actual model parameters
        print("Received global model broadcast")
    
    def _handle_training_request(self, request_data: Dict[str, Any]):
        """Handle training request"""
        training_mode = TrainingMode(request_data.get('mode', 'individual'))
        
        if self.current_state == VehicleState.IDLE:
            can_participate, reason = self.vehicle_monitor.can_participate_in_training()
            
            if can_participate:
                self.current_state = VehicleState.AGGREGATING if training_mode == TrainingMode.INDIVIDUAL else VehicleState.PIPELINE
                self.training_mode = training_mode
                
                # Start training execution
                self._start_training_session(training_mode, request_data)
            else:
                print(f"Cannot participate in training: {reason}")
    
    def _handle_error_propagation(self, error_data: Dict[str, Any]):
        """Handle error propagation"""
        # Process error signals
        print("Received error propagation data")
    
    def _start_training_session(self, training_mode: TrainingMode, request_data: Dict[str, Any]):
        """Start training session"""
        try:
            # Create mock model and data for demonstration
            model = self._create_mock_model()
            training_data = self._create_mock_data()
            
            from core.types import TrainingConfig, TrainingTask
            training_config = TrainingConfig(
                epochs=request_data.get('epochs', 1),
                batch_size=request_data.get('batch_size', 32),
                learning_rate=request_data.get('learning_rate', 0.001)
            )
            
            task = TrainingTask(
                task_id=f"task_{self.vehicle_info.vehicle_id}_{int(time.time())}",
                model=model,
                training_data=training_data,
                config=training_config,
                pipeline_id=self.active_pipeline if training_mode == TrainingMode.PIPELINE else None
            )
            
            # Submit training task
            self.training_executor.submit_training_task(task)
            
            # Update statistics
            self.vehicle_stats['total_training_sessions'] += 1
            if training_mode == TrainingMode.PIPELINE:
                self.vehicle_stats['pipeline_participations'] += 1
            else:
                self.vehicle_stats['individual_participations'] += 1
            
            print(f"Started {training_mode.value} training session")
            
        except Exception as e:
            print(f"Failed to start training session: {e}")
            self.current_state = VehicleState.IDLE
    
    def _create_mock_model(self) -> nn.Module:
        """Create mock neural network model"""
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def _create_mock_data(self):
        """Create mock training data"""
        # Create dummy data for demonstration
        class MockDataLoader:
            def __init__(self):
                self.dataset = [(torch.randn(784), torch.randint(0, 10, (1,)).squeeze()) for _ in range(100)]
                
            def __iter__(self):
                batch_size = 32
                for i in range(0, len(self.dataset), batch_size):
                    batch = self.dataset[i:i+batch_size]
                    if len(batch) < batch_size:
                        continue
                    data = torch.stack([item[0] for item in batch])
                    target = torch.stack([item[1] for item in batch])
                    yield data, target
                    
        return MockDataLoader()
    
    def initiate_pipeline_formation(self, target_vehicles: List[str]):
        """Initiate pipeline formation with target vehicles"""
        try:
            # Get best neighbors for pipeline
            best_neighbors = self.communication_manager.get_best_neighbors(len(target_vehicles))
            
            if len(best_neighbors) >= 2:  # Need at least 2 for pipeline
                pipeline_info = {
                    'initiator': self.vehicle_info.vehicle_id,
                    'target_vehicles': [vid for vid, _ in best_neighbors],
                    'template': self._create_pipeline_template(len(best_neighbors))
                }
                
                self.communication_manager.send_pipeline_invitation(
                    [vid for vid, _ in best_neighbors], pipeline_info
                )
                
                print(f"Initiated pipeline formation with {len(best_neighbors)} vehicles")
            else:
                print("Insufficient neighbors for pipeline formation")
                
        except Exception as e:
            print(f"Pipeline formation initiation failed: {e}")
    
    def _create_pipeline_template(self, num_vehicles: int) -> Dict[str, Any]:
        """Create pipeline template for formation"""
        from core.types import ResourceClass
        
        return {
            'template_id': f"template_{self.vehicle_info.vehicle_id}_{int(time.time())}",
            'resource_requirements': [ResourceClass.MEDIUM.value] * num_vehicles,
            'expected_duration': 15.0,
            'communication_pattern': [(i, i+1) for i in range(num_vehicles-1)]
        }
    
    def submit_model_update(self, update_data: torch.Tensor, metadata: Dict[str, Any]):
        """Submit model update to neighbors or edge server"""
        try:
            from core.types import ModelUpdate
            update = ModelUpdate(
                source_id=self.vehicle_info.vehicle_id,
                update_data=update_data,
                metadata=metadata,
                training_mode=self.training_mode or TrainingMode.INDIVIDUAL
            )
            
            # Prepare communication
            communication_data = self.training_executor.prepare_communication(update)
            
            if isinstance(communication_data, dict):
                # Single message
                if self.active_pipeline:
                    # Send to pipeline neighbors
                    pass  # In real implementation, send to specific targets
                else:
                    # Send to edge server or broadcast
                    pass
            else:
                # Communication bundle
                pass  # Handle bundle communication
            
            self.vehicle_stats['messages_sent'] += 1
            
        except Exception as e:
            print(f"Model update submission failed: {e}")
    
    def get_vehicle_status(self) -> Dict[str, Any]:
        """Get comprehensive vehicle status"""
        current_time = time.time()
        uptime = current_time - self.start_time if hasattr(self, 'start_time') else 0
        
        return {
            'vehicle_id': self.vehicle_info.vehicle_id,
            'current_state': self.current_state.value,
            'active_pipeline': self.active_pipeline,
            'training_mode': self.training_mode.value if self.training_mode else None,
            'position': self.vehicle_info.position,
            'velocity': self.vehicle_info.velocity,
            'direction': self.vehicle_info.direction,
            'neighbors_count': len(self.neighbors),
            'vehicle_active': self.vehicle_active,
            'uptime_seconds': uptime,
            'monitor_status': self.vehicle_monitor.get_vehicle_status(),
            'training_stats': self.training_executor.get_execution_statistics(),
            **self.vehicle_stats
        }
    
    def set_training_mode(self, mode: TrainingMode):
        """Set training mode"""
        self.training_mode = mode
    
    def enter_idle_state(self):
        """Enter idle state"""
        self.current_state = VehicleState.IDLE
        self.active_pipeline = None
        self.training_mode = None