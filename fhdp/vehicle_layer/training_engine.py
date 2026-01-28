"""
Training Execution Engine with Communication Optimization for FHDP System

Implements training execution with short-horizon training (1-2 epochs),
communication optimization through bundling and compression, and
lazy error propagation to reduce communication overhead.
"""
import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
import numpy as np
import pickle
import zlib
import json

from core.types import (
    VehicleInfo, Pipeline, PipelineTemplate, ModelUpdate, TrainingMode,
    TrainingConfig, CommunicationBundle, ErrorPropagation, ResourceMetrics
)
from core.constants import (
    TRAINING_EPOCHS_SHORT, COMMUNICATION_BUNDLE_SIZE,
    ERROR_ACCUMULATION_THRESHOLD, MAX_ERROR_PROPAGATION_DELAY,
    MAX_VEHICLE_MEMORY_USAGE, MIN_LOCAL_DATA_SIZE
)

@dataclass
class TrainingTask:
    """Training task definition"""
    task_id: str
    model: nn.Module
    training_data: torch.utils.data.DataLoader
    config: TrainingConfig
    pipeline_id: Optional[str] = None
    position_in_pipeline: int = -1
    target_vehicle_id: Optional[str] = None  # For pipeline training
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    status: str = "pending"  # pending, running, completed, failed

@dataclass
class TrainingResult:
    """Training execution result"""
    task_id: str
    model_update: torch.Tensor
    metadata: Dict[str, Any]
    training_time: float
    data_size: int
    loss: float
    accuracy: Optional[float] = None
    error_signal: Optional[torch.Tensor] = None

class CommunicationOptimizer:
    """Optimizes communication through compression and bundling"""
    
    def __init__(self):
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0.0,
            'compression_time': 0.0
        }
        self.bundle_buffer: List[Dict[str, Any]] = []
        self.bundle_buffer_size = 0
        self.last_bundle_time = time.time()
        
    def compress_model_update(self, model_update: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                            compression_level: int = 6) -> Tuple[bytes, float]:
        """Compress model update for transmission"""
        start_time = time.time()
        
        # Serialize tensor data
        if isinstance(model_update, dict):
            # Dictionary of tensors
            serialized = {}
            for key, tensor in model_update.items():
                serialized[key] = {
                    'data': tensor.detach().cpu().numpy().tolist(),
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype)
                }
        else:
            # Single tensor
            serialized = {
                'data': model_update.detach().cpu().numpy().tolist(),
                'shape': model_update.shape,
                'dtype': str(model_update.dtype)
            }
        
        # Convert to JSON string
        json_str = json.dumps(serialized)
        original_bytes = json_str.encode('utf-8')
        
        # Compress using zlib
        compressed_bytes = zlib.compress(original_bytes, compression_level)
        
        compression_time = time.time() - start_time
        
        # Update statistics
        self.compression_stats['original_size'] += len(original_bytes)
        self.compression_stats['compressed_size'] += len(compressed_bytes)
        total_ratio = (self.compression_stats['compressed_size'] / 
                      max(1, self.compression_stats['original_size']))
        self.compression_stats['compression_ratio'] = 1.0 - total_ratio
        self.compression_stats['compression_time'] += compression_time
        
        return compressed_bytes, len(original_bytes) / len(compressed_bytes)
    
    def decompress_model_update(self, compressed_data: bytes) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Decompress model update"""
        # Decompress data
        json_str = zlib.decompress(compressed_data).decode('utf-8')
        serialized = json.loads(json_str)
        
        if 'data' in serialized and not isinstance(serialized['data'], list):
            # Single tensor
            return torch.tensor(serialized['data']).reshape(serialized['shape'])
        else:
            # Dictionary of tensors
            result = {}
            for key, value in serialized.items():
                if isinstance(value, dict) and 'data' in value:
                    result[key] = torch.tensor(value['data']).reshape(value['shape'])
            return result
    
    def create_bundle(self, messages: List[Dict[str, Any]], target_protocol: str = "dsrc") -> CommunicationBundle:
        """Create communication bundle for efficient transmission"""
        # Calculate bundle size before compression
        bundle_messages = []
        total_size = 0
        
        for message in messages:
            compressed_data, compression_ratio = self.compress_model_update(
                message.get('model_update', {})
            )
            
            bundle_message = {
                'message_id': message['message_id'],
                'sender_id': message['sender_id'],
                'receiver_id': message['receiver_id'],
                'message_type': message['message_type'],
                'compressed_data': compressed_data,
                'metadata': message.get('metadata', {}),
                'compression_ratio': compression_ratio,
                'timestamp': message['timestamp']
            }
            
            bundle_messages.append(bundle_message)
            total_size += len(compressed_data)
        
        bundle = CommunicationBundle(
            messages=bundle_messages,
            target_ids=[msg['receiver_id'] for msg in bundle_messages],
            protocol=target_protocol,
            compression_ratio=np.mean([msg['compression_ratio'] for msg in bundle_messages]),
            bundle_size=total_size
        )
        
        return bundle
    
    def add_to_bundle_buffer(self, message: Dict[str, Any]) -> bool:
        """Add message to bundle buffer"""
        message_size = len(str(message))
        
        # Check if adding this message would exceed buffer size
        if self.bundle_buffer_size + message_size > COMMUNICATION_BUNDLE_SIZE:
            return False
        
        self.bundle_buffer.append(message)
        self.bundle_buffer_size += message_size
        return True
    
    def should_flush_buffer(self) -> bool:
        """Determine if buffer should be flushed"""
        current_time = time.time()
        
        # Flush if buffer is full or timeout reached
        buffer_full = self.bundle_buffer_size >= COMMUNICATION_BUNDLE_SIZE
        timeout_reached = (current_time - self.last_bundle_time) > 5.0  # 5 second timeout
        
        return buffer_full or timeout_reached
    
    def flush_buffer(self, protocol: str = "dsrc") -> Optional[CommunicationBundle]:
        """Flush buffer and create bundle"""
        if not self.bundle_buffer:
            return None
        
        bundle = self.create_bundle(self.bundle_buffer, protocol)
        
        # Clear buffer
        self.bundle_buffer.clear()
        self.bundle_buffer_size = 0
        self.last_bundle_time = time.time()
        
        return bundle

class LazyErrorPropagation:
    """Implements lazy error propagation to reduce communication overhead"""
    
    def __init__(self):
        self.error_accumulator: Dict[str, torch.Tensor] = {}
        self.error_threshold = ERROR_ACCUMULATION_THRESHOLD
        self.last_propagation = time.time()
        self.propagation_targets: List[str] = []
        self.propagation_count = 0
        
    def accumulate_error(self, source_id: str, error_signal: torch.Tensor):
        """Accumulate error signal from training"""
        if source_id in self.error_accumulator:
            self.error_accumulator[source_id] += error_signal
        else:
            self.error_accumulator[source_id] = error_signal.clone()
    
    def should_propagate_errors(self) -> bool:
        """Determine if accumulated errors should be propagated"""
        current_time = time.time()
        
        # Check time threshold
        time_threshold = (current_time - self.last_propagation) > MAX_ERROR_PROPAGATION_DELAY
        
        # Check accumulation threshold
        total_error_magnitude = sum(
            torch.norm(error).item() for error in self.error_accumulator.values()
        )
        error_threshold = total_error_magnitude >= self.error_threshold
        
        # Check count threshold (propagate every N accumulations)
        count_threshold = self.propagation_count >= 5
        
        return time_threshold or error_threshold or count_threshold
    
    def get_propagation_errors(self) -> ErrorPropagation:
        """Get errors ready for propagation"""
        errors_to_propagate = ErrorPropagation(
            error_signals=self.error_accumulator.copy(),
            accumulation_threshold=self.error_threshold,
            propagation_targets=self.propagation_targets.copy(),
            propagation_count=self.propagation_count
        )
        
        # Reset accumulator
        self.error_accumulator.clear()
        self.last_propagation = time.time()
        self.propagation_count += 1
        
        return errors_to_propagate

class TrainingExecutor:
    """Executes training tasks with resource constraints"""
    
    def __init__(self, vehicle_info: VehicleInfo):
        self.vehicle_info = vehicle_info
        self.communication_optimizer = CommunicationOptimizer()
        self.error_propagation = LazyErrorPropagation()
        
        # Training state
        self.active_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: Dict[str, TrainingResult] = {}
        self.training_queue = deque()
        
        # Resource monitoring
        self.resource_usage = ResourceMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            battery_level=vehicle_info.resources.get('battery', 1.0),
            network_quality=vehicle_info.resources.get('network_quality', 1.0),
            thermal_state=0.0
        )
        
        # Threading
        self.executor_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.execution_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0,
            'total_data_processed': 0
        }
    
    def start_execution_service(self):
        """Start training execution service"""
        if self.executor_thread:
            return
        
        self.stop_event.clear()
        self.executor_thread = threading.Thread(target=self._execution_worker, daemon=True)
        self.executor_thread.start()
    
    def stop_execution_service(self):
        """Stop training execution service"""
        self.stop_event.set()
        
        if self.executor_thread:
            self.executor_thread.join(timeout=5.0)
    
    def submit_training_task(self, task: TrainingTask) -> str:
        """Submit training task for execution"""
        self.active_tasks[task.task_id] = task
        self.training_queue.append(task.task_id)
        self.execution_stats['total_tasks'] += 1
        
        return task.task_id
    
    def _execution_worker(self):
        """Training execution worker thread"""
        while not self.stop_event.is_set():
            try:
                if self.training_queue:
                    task_id = self.training_queue.popleft()
                    task = self.active_tasks.get(task_id)
                    
                    if task:
                        # Check resource availability
                        if self._can_execute_task(task):
                            result = self._execute_training_task(task)
                            self._handle_training_result(task, result)
                        else:
                            # Re-queue task if resources unavailable
                            self.training_queue.append(task_id)
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"Execution worker error: {e}")
    
    def _can_execute_task(self, task: TrainingTask) -> bool:
        """Check if task can be executed with available resources"""
        # Check memory usage
        if self.resource_usage.memory_usage > MAX_VEHICLE_MEMORY_USAGE:
            return False
        
        # Check battery level
        if self.resource_usage.battery_level < 0.2:  # 20% minimum
            return False
        
        # Check thermal state
        if self.resource_usage.thermal_state > 0.8:  # 80% thermal threshold
            return False
        
        return True
    
    def _execute_training_task(self, task: TrainingTask) -> Optional[TrainingResult]:
        """Execute training task"""
        try:
            start_time = time.time()
            task.status = "running"
            
            # Update resource usage
            self._update_resource_usage(task)
            
            # Configure training
            model = task.model
            optimizer = optim.SGD(model.parameters(), lr=task.config.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Short-horizon training (1-2 epochs)
            num_epochs = np.random.randint(*TRAINING_EPOCHS_SHORT) if task.config.epochs == 0 else task.config.epochs
            
            model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(task.training_data):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Accumulate statistics
                    epoch_loss += loss.item()
                    if hasattr(output, 'argmax'):
                        pred = output.argmax(dim=1, keepdim=True)
                        correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                    total_samples += target.size(0)
                    
                    # Memory cleanup
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                total_loss += epoch_loss
            
            # Calculate model update (difference from initial weights)
            model_update = self._calculate_model_update(model)
            
            # Calculate error signal for lazy propagation
            error_signal = torch.tensor(total_loss / len(task.training_data))
            
            execution_time = time.time() - start_time
            
            # Create result
            result = TrainingResult(
                task_id=task.task_id,
                model_update=model_update,
                metadata={
                    'epochs': num_epochs,
                    'batch_size': task.config.batch_size,
                    'learning_rate': task.config.learning_rate,
                    'pipeline_id': task.pipeline_id,
                    'position_in_pipeline': task.position_in_pipeline,
                    'data_size': len(task.training_data.dataset)
                },
                training_time=execution_time,
                data_size=len(task.training_data.dataset),
                loss=total_loss / num_epochs,
                accuracy=correct_predictions / max(1, total_samples),
                error_signal=error_signal
            )
            
            # Update statistics
            self.execution_stats['completed_tasks'] += 1
            self.execution_stats['total_data_processed'] += len(task.training_data.dataset)
            
            return result
            
        except Exception as e:
            print(f"Training execution failed: {e}")
            task.status = "failed"
            self.execution_stats['failed_tasks'] += 1
            return None
    
    def _calculate_model_update(self, model: nn.Module) -> torch.Tensor:
        """Calculate model update (parameter differences)"""
        # In real implementation, this would compute difference from received model
        # For now, return current parameters as update
        params = []
        for param in model.parameters():
            params.append(param.data.flatten())
        
        return torch.cat(params)
    
    def _update_resource_usage(self, task: TrainingTask):
        """Update resource usage during training"""
        # Estimate CPU usage
        self.resource_usage.cpu_usage = min(1.0, self.resource_usage.cpu_usage + 0.3)
        
        # Estimate memory usage
        self.resource_usage.memory_usage = min(1.0, self.resource_usage.memory_usage + 0.2)
        
        # Estimate battery consumption
        self.resource_usage.battery_level = max(0.0, self.resource_usage.battery_level - 0.001)
        
        # Estimate thermal increase
        self.resource_usage.thermal_state = min(1.0, self.resource_usage.thermal_state + 0.01)
    
    def _handle_training_result(self, task: TrainingTask, result: Optional[TrainingResult]):
        """Handle training completion result"""
        if result:
            # Move task to completed
            del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = result
            
            # Accumulate error for lazy propagation
            if result.error_signal is not None:
                self.error_propagation.accumulate_error(task.task_id, result.error_signal)
            
            # Update average execution time
            total_completed = self.execution_stats['completed_tasks']
            old_avg = self.execution_stats['avg_execution_time']
            self.execution_stats['avg_execution_time'] = (
                (old_avg * (total_completed - 1) + result.training_time) / total_completed
            )
        else:
            # Task failed
            task.status = "failed"
    
    def create_model_update(self, result: TrainingResult, training_mode: TrainingMode) -> ModelUpdate:
        """Create model update from training result"""
        return ModelUpdate(
            source_id=self.vehicle_info.vehicle_id,
            update_data=result.model_update,
            metadata=result.metadata,
            timestamp=time.time(),
            training_mode=training_mode,
            fidelity_score=self._calculate_fidelity_score(result)
        )
    
    def _calculate_fidelity_score(self, result: TrainingResult) -> float:
        """Calculate fidelity score for model update"""
        # Base fidelity on training quality
        loss_factor = max(0.0, 1.0 - result.loss)  # Lower loss = higher fidelity
        accuracy_factor = result.accuracy if result.accuracy else 0.5
        
        # Factor in resource constraints
        resource_factor = 1.0 - (self.resource_usage.cpu_usage + self.resource_usage.memory_usage) / 2.0
        
        fidelity = (loss_factor * 0.4 + accuracy_factor * 0.4 + resource_factor * 0.2)
        return max(0.1, min(1.0, fidelity))
    
    def prepare_communication(self, model_update: ModelUpdate) -> Union[CommunicationBundle, Dict[str, Any]]:
        """Prepare model update for communication with optimization"""
        message = {
            'message_id': f"update_{model_update.source_id}_{int(time.time() * 1000)}",
            'sender_id': model_update.source_id,
            'receiver_id': model_update.metadata.get('target_vehicle_id', 'broadcast'),
            'message_type': 'model_update',
            'model_update': model_update.update_data,
            'metadata': model_update.metadata,
            'timestamp': model_update.timestamp
        }
        
        # Try to add to bundle buffer
        if self.communication_optimizer.add_to_bundle_buffer(message):
            # Check if buffer should be flushed
            if self.communication_optimizer.should_flush_buffer():
                bundle = self.communication_optimizer.flush_buffer()
                if bundle:
                    return bundle
        
        # Return single message
        return message
    
    def get_error_propagation_data(self) -> Optional[ErrorPropagation]:
        """Get error propagation data if ready"""
        if self.error_propagation.should_propagate_errors():
            return self.error_propagation.get_propagation_errors()
        return None
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get training execution statistics"""
        return {
            **self.execution_stats,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'queue_length': len(self.training_queue),
            'resource_usage': {
                'cpu': self.resource_usage.cpu_usage,
                'memory': self.resource_usage.memory_usage,
                'battery': self.resource_usage.battery_level,
                'thermal': self.resource_usage.thermal_state
            },
            'communication_stats': self.communication_optimizer.compression_stats
        }
    
    def update_resources(self, resource_metrics: ResourceMetrics):
        """Update resource usage metrics"""
        self.resource_usage = resource_metrics