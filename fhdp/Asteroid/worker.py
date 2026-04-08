"""Asteroid Worker for executing HPP training"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import queue
from .scheduler import MicroBatchScheduler
from .fault_tolerance import FaultToleranceManager

class InMemoryTaskPool:
    """In-memory task pool for FP/BP tasks"""
    
    def __init__(self):
        """
        Initialize the task pool
        """
        self.tasks = queue.Queue()
        self.completed_tasks = []
    
    def add_task(self, task: Dict):
        """
        Add a task to the pool
        
        Args:
            task: Task dictionary
        """
        self.tasks.put(task)
    
    def get_task(self) -> Optional[Dict]:
        """
        Get a task from the pool
        
        Returns:
            Task dictionary or None if pool is empty
        """
        try:
            return self.tasks.get(block=False)
        except queue.Empty:
            return None
    
    def mark_completed(self, task: Dict):
        """
        Mark a task as completed
        
        Args:
            task: Completed task
        """
        self.completed_tasks.append(task)
    
    def get_completed(self) -> List[Dict]:
        """
        Get completed tasks
        
        Returns:
            List of completed tasks
        """
        return self.completed_tasks

class ModelExecutor:
    """Model executor for running FP/BP tasks"""
    
    def __init__(self, model: nn.Module, device: str):
        """
        Initialize the model executor
        
        Args:
            model: PyTorch model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(input_tensor)
    
    def backward(self, output_tensor: torch.Tensor, target_tensor: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """
        Run backward pass
        
        Args:
            output_tensor: Output tensor from forward pass
            target_tensor: Target tensor
            loss_fn: Loss function
            
        Returns:
            Loss value
        """
        loss = loss_fn(output_tensor, target_tensor)
        loss.backward()
        return loss

class TensorDispatcher:
    """Tensor dispatcher for sending activations/gradients"""
    
    def __init__(self):
        """
        Initialize the tensor dispatcher
        """
        self.connections = {}
    
    def send_tensor(self, tensor: torch.Tensor, destination: str) -> bool:
        """
        Send tensor to destination
        
        Args:
            tensor: Tensor to send
            destination: Destination device
            
        Returns:
            True if send was successful
        """
        # This is a simplified implementation
        # In practice, you would use inter-device communication
        print(f"[TensorDispatcher] Sending tensor to {destination}")
        return True
    
    def receive_tensor(self, source: str) -> Optional[torch.Tensor]:
        """
        Receive tensor from source
        
        Args:
            source: Source device
            
        Returns:
            Received tensor or None
        """
        # This is a simplified implementation
        # In practice, you would use inter-device communication
        print(f"[TensorDispatcher] Receiving tensor from {source}")
        return None

class AsteroidWorker:
    """Asteroid Worker for executing HPP training"""
    
    def __init__(self, model: nn.Module, device: str, fault_tolerance: FaultToleranceManager = None):
        """
        Initialize the worker
        
        Args:
            model: PyTorch model
            device: Device to run on
            fault_tolerance: Fault tolerance manager
        """
        self.model = model
        self.device = device
        self.task_pool = InMemoryTaskPool()
        self.scheduler = MicroBatchScheduler(strategy="1F1B")
        self.executor = ModelExecutor(model, device)
        self.dispatcher = TensorDispatcher()
        self.fault_tolerance = fault_tolerance
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """
        Start the worker
        """
        self.running = True
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """
        Stop the worker
        """
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def add_task(self, task: Dict):
        """
        Add a task to the task pool
        
        Args:
            task: Task dictionary
        """
        self.task_pool.add_task(task)
    
    def _run(self):
        """
        Run the worker loop
        """
        while self.running:
            # Get next task
            task = self.task_pool.get_task()
            if not task:
                continue
            
            # Execute task
            try:
                if task['type'] == 'forward':
                    # Forward pass
                    input_tensor = task.get('input')
                    if input_tensor is not None:
                        output = self.executor.forward(input_tensor)
                        # Send output to next stage
                        if 'next_stage' in task:
                            self.dispatcher.send_tensor(output, task['next_stage'])
                elif task['type'] == 'backward':
                    # Backward pass
                    output_tensor = task.get('output')
                    target_tensor = task.get('target')
                    if output_tensor is not None and target_tensor is not None:
                        loss = self.executor.backward(output_tensor, target_tensor, self.loss_fn)
                        # Send gradient to previous stage
                        if 'prev_stage' in task:
                            # Get gradients
                            gradients = [param.grad for param in self.model.parameters() if param.grad is not None]
                            if gradients:
                                # Send gradients
                                self.dispatcher.send_tensor(gradients[0], task['prev_stage'])
            except Exception as e:
                print(f"[Worker] Error executing task: {e}")
            
            # Mark task as completed
            self.task_pool.mark_completed(task)
            
            # Update heartbeat if fault tolerance is enabled
            if self.fault_tolerance:
                self.fault_tolerance.update_heartbeat(self.device)
    
    def run_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage_index: int, num_stages: int):
        """
        Run a training step
        
        Args:
            batch: Input batch (inputs, targets)
            stage_index: Current stage index
            num_stages: Total number of stages
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Generate schedule
        num_micro_batches = 4  # Example value
        schedule = self.scheduler.generate_schedule(num_micro_batches, num_stages)
        
        # Process schedule
        for task in schedule:
            if task['stage'] == stage_index:
                if task['type'] == 'forward':
                    # Create forward task
                    forward_task = {
                        'type': 'forward',
                        'micro_batch': task['micro_batch'],
                        'stage': stage_index,
                        'input': inputs,
                        'next_stage': f'stage_{stage_index + 1}' if stage_index < num_stages - 1 else None
                    }
                    self.add_task(forward_task)
                elif task['type'] == 'backward':
                    # Create backward task
                    backward_task = {
                        'type': 'backward',
                        'micro_batch': task['micro_batch'],
                        'stage': stage_index,
                        'target': targets,
                        'prev_stage': f'stage_{stage_index - 1}' if stage_index > 0 else None
                    }
                    self.add_task(backward_task)
    
    def update_weights(self):
        """
        Update model weights
        """
        self.optimizer.step()
        self.optimizer.zero_grad()
