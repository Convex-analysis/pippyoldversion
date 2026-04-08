"""Micro-batch scheduler for Asteroid"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import queue

class MicroBatchScheduler:
    """Micro-batch scheduler for 1F1B scheduling"""
    
    def __init__(self, strategy: str = "1F1B"):
        """
        Initialize the scheduler
        
        Args:
            strategy: Scheduling strategy ("1F1B", "Gpipe")
        """
        self.strategy = strategy
        self.task_queue = queue.Queue()
        self.completed_tasks = []
    
    def generate_schedule(self, num_micro_batches: int, num_stages: int) -> List[Dict]:
        """
        Generate micro-batch schedule
        
        Args:
            num_micro_batches: Number of micro-batches
            num_stages: Number of pipeline stages
            
        Returns:
            List of scheduled tasks
        """
        if self.strategy == "1F1B":
            return self._generate_1f1b_schedule(num_micro_batches, num_stages)
        elif self.strategy == "Gpipe":
            return self._generate_gpipe_schedule(num_micro_batches, num_stages)
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.strategy}")
    
    def _generate_1f1b_schedule(self, num_micro_batches: int, num_stages: int) -> List[Dict]:
        """
        Generate 1F1B schedule
        
        Args:
            num_micro_batches: Number of micro-batches
            num_stages: Number of pipeline stages
            
        Returns:
            List of scheduled tasks
        """
        schedule = []
        M = num_micro_batches
        P = num_stages
        
        # 1F1B schedule: process forward pass for micro-batch i, then backward pass for micro-batch i-P+1
        for t in range(M + P - 1):
            for p in range(P):
                i = t - p
                if 0 <= i < M:
                    # Forward pass
                    schedule.append({
                        'type': 'forward',
                        'micro_batch': i,
                        'stage': p,
                        'time_step': t
                    })
                
                j = t - (P - 1 - p)
                if 0 <= j < M:
                    # Backward pass
                    schedule.append({
                        'type': 'backward',
                        'micro_batch': j,
                        'stage': p,
                        'time_step': t
                    })
        
        return schedule
    
    def _generate_gpipe_schedule(self, num_micro_batches: int, num_stages: int) -> List[Dict]:
        """
        Generate Gpipe schedule (backward-after-forward)
        
        Args:
            num_micro_batches: Number of micro-batches
            num_stages: Number of pipeline stages
            
        Returns:
            List of scheduled tasks
        """
        schedule = []
        M = num_micro_batches
        P = num_stages
        
        # Gpipe schedule: process all forward passes first, then all backward passes
        for i in range(M):
            for p in range(P):
                schedule.append({
                    'type': 'forward',
                    'micro_batch': i,
                    'stage': p,
                    'time_step': i * P + p
                })
        
        for i in range(M):
            for p in reversed(range(P)):
                schedule.append({
                    'type': 'backward',
                    'micro_batch': i,
                    'stage': p,
                    'time_step': M * P + i * P + (P - 1 - p)
                })
        
        return schedule
    
    def add_task(self, task: Dict):
        """
        Add a task to the queue
        
        Args:
            task: Task dictionary
        """
        self.task_queue.put(task)
    
    def get_next_task(self) -> Optional[Dict]:
        """
        Get the next task from the queue
        
        Returns:
            Next task or None if queue is empty
        """
        try:
            return self.task_queue.get(block=False)
        except queue.Empty:
            return None
    
    def mark_task_completed(self, task: Dict):
        """
        Mark a task as completed
        
        Args:
            task: Completed task
        """
        self.completed_tasks.append(task)
    
    def get_completed_tasks(self) -> List[Dict]:
        """
        Get completed tasks
        
        Returns:
            List of completed tasks
        """
        return self.completed_tasks
    
    def reset(self):
        """
        Reset the scheduler
        """
        self.task_queue = queue.Queue()
        self.completed_tasks = []
