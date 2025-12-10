"""
Asynchronous Aggregation Engine for FHDP System

Handles asynchronous federated aggregation of model updates from both individual
vehicles and pipeline training sessions. Maintains global model state and
ensures efficient, fair aggregation with minimal communication overhead.
"""
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import queue
import torch

from ..core.types import (
    ModelUpdate, AggregationResult, TrainingMode, FairnessMetrics,
    VehicleInfo, ErrorPropagation
)
from ..core.constants import (
    ASYNC_AGGREGATION_INTERVAL, MIN_AGGREGATION_PARTICIPANTS,
    AGGREGATION_TIMEOUT, WEIGHT_DECAY_FACTOR, 
    ERROR_ACCUMULATION_THRESHOLD, MAX_ERROR_PROPAGATION_DELAY
)

@dataclass
class AggregationBuffer:
    """Buffer for managing asynchronous model updates"""
    updates: deque = field(default_factory=lambda: deque(maxlen=1000))
    weights: Dict[str, float] = field(default_factory=dict)
    last_aggregation: float = field(default_factory=time.time)
    pending_count: int = 0
    
class WeightCalculator:
    """Calculates aggregation weights based on various factors"""
    
    def __init__(self):
        self.contribution_history = defaultdict(list)
        self.fairness_weights = defaultdict(float)
        
    def calculate_weight(self, update: ModelUpdate, fairness_metrics: Optional[FairnessMetrics] = None) -> float:
        """Calculate aggregation weight for model update"""
        base_weight = 1.0
        
        # Data size factor
        data_size = update.metadata.get('data_size', 1000)
        size_factor = data_size / 1000.0  # Normalize to 1KB
        base_weight *= size_factor
        
        # Fidelity factor
        fidelity_factor = update.fidelity_score
        base_weight *= fidelity_factor
        
        # Freshness factor (newer updates get higher weight)
        age_factor = np.exp(-(time.time() - update.timestamp) / 10.0)  # 10-second decay
        base_weight *= age_factor
        
        # Training mode factor
        if update.training_mode == TrainingMode.PIPELINE:
            # Pipeline updates might be more reliable due to collaboration
            base_weight *= 1.2
        
        # Fairness factor
        if fairness_metrics:
            fairness_factor = fairness_metrics.priority_weight
            base_weight *= fairness_factor
            
            # Update contribution history
            self.contribution_history[update.source_id].append((time.time(), base_weight))
            
            # Maintain sliding window of contributions
            if len(self.contribution_history[update.source_id]) > 100:
                self.contribution_history[update.source_id].popleft()
        
        return base_weight
    
    def update_fairness_weights(self, metrics: Dict[str, FairnessMetrics]):
        """Update fairness weights based on participation metrics"""
        current_time = time.time()
        
        for vehicle_id, metric in metrics.items():
            # Calculate recency penalty
            time_since_last = current_time - metric.last_participation
            recency_penalty = min(2.0, 1.0 + time_since_last / 60.0)  # Max 2x weight
            
            # Calculate participation frequency
            participation_rate = metric.participation_count / max(1.0, time_since_last / 3600.0)  # per hour
            
            # Combine factors
            self.fairness_weights[vehicle_id] = recency_penalty * metric.contribution_score
            
            # Penalize over-participation
            if participation_rate > 5.0:  # More than 5 participations per hour
                self.fairness_weights[vehicle_id] *= 0.5

class AsynchronousAggregator:
    """Main asynchronous aggregation engine"""
    
    def __init__(self, initial_model: Optional[Dict[str, torch.Tensor]] = None):
        self.global_model = initial_model or {}
        self.aggregation_buffer = AggregationBuffer()
        self.weight_calculator = WeightCalculator()
        
        # Threading for async processing
        self.aggregation_thread = None
        self.stop_event = threading.Event()
        self.update_queue = queue.Queue()
        
        # Error propagation management
        self.error_accumulator = defaultdict(lambda: torch.zeros(1))  # Will be properly initialized
        self.error_propagation_queue = queue.Queue()
        
        # Aggregation statistics
        self.aggregation_stats = {
            'total_aggregations': 0,
            'successful_aggregations': 0,
            'avg_participants': 0.0,
            'total_updates_processed': 0
        }
        
        # Start background aggregation thread
        self._start_aggregation_thread()
        self._start_error_propagation_thread()
    
    def _start_aggregation_thread(self):
        """Start background aggregation thread"""
        def aggregation_worker():
            while not self.stop_event.is_set():
                try:
                    # Wait for updates or timeout
                    self.update_queue.get(timeout=ASYNC_AGGREGATION_INTERVAL)
                    
                    # Check if we should aggregate
                    if self._should_aggregate():
                        self._perform_aggregation()
                        
                except queue.Empty:
                    # Timeout occurred, check if we should aggregate anyway
                    if self._should_aggregate(force=True):
                        self._perform_aggregation()
                except Exception as e:
                    print(f"Aggregation error: {e}")
        
        self.aggregation_thread = threading.Thread(target=aggregation_worker, daemon=True)
        self.aggregation_thread.start()
    
    def _start_error_propagation_thread(self):
        """Start error propagation thread"""
        def error_worker():
            while not self.stop_event.is_set():
                try:
                    error_data = self.error_propagation_queue.get(timeout=MAX_ERROR_PROPAGATION_DELAY)
                    
                    # Check accumulation threshold
                    if self._should_propagate_errors(error_data):
                        self._propagate_errors(error_data)
                        
                except queue.Empty:
                    # Check for accumulated errors
                    if len(self.error_accumulator) > 0:
                        self._propagate_accumulated_errors()
                except Exception as e:
                    print(f"Error propagation error: {e}")
        
        error_thread = threading.Thread(target=error_worker, daemon=True)
        error_thread.start()
    
    def submit_update(self, update: ModelUpdate, fairness_metrics: Optional[FairnessMetrics] = None):
        """Submit model update for asynchronous aggregation"""
        try:
            # Calculate weight
            weight = self.weight_calculator.calculate_weight(update, fairness_metrics)
            
            # Add to buffer
            self.aggregation_buffer.updates.append(update)
            self.aggregation_buffer.weights[update.source_id] = weight
            self.aggregation_buffer.pending_count += 1
            
            # Notify aggregation thread
            self.update_queue.put(update)
            
            # Update statistics
            self.aggregation_stats['total_updates_processed'] += 1
            
        except Exception as e:
            print(f"Error submitting update: {e}")
    
    def _should_aggregate(self, force: bool = False) -> bool:
        """Determine if aggregation should be performed"""
        current_time = time.time()
        
        # Force aggregation if timeout reached
        if force:
            return self.aggregation_buffer.pending_count >= MIN_AGGREGATION_PARTICIPANTS
        
        # Check time since last aggregation
        time_since_last = current_time - self.aggregation_buffer.last_aggregation
        if time_since_last >= AGGREGATION_TIMEOUT:
            return self.aggregation_buffer.pending_count >= MIN_AGGREGATION_PARTICIPANTS
        
        # Check buffer size
        return self.aggregation_buffer.pending_count >= MIN_AGGREGATION_PARTICIPANTS * 2
    
    def _perform_aggregation(self):
        """Perform federated aggregation"""
        if not self.aggregation_buffer.updates:
            return
        
        try:
            # Extract updates and weights
            updates_to_process = list(self.aggregation_buffer.updates)
            weights = {u.source_id: self.aggregation_buffer.weights.get(u.source_id, 1.0) 
                      for u in updates_to_process}
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight == 0:
                return
            
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            
            # Perform weighted aggregation
            aggregated_model = self._aggregate_models(updates_to_process, normalized_weights)
            
            if aggregated_model:
                # Update global model
                self.global_model = aggregated_model
                
                # Create aggregation result
                result = AggregationResult(
                    global_model=aggregated_model,
                    participating_sources=[u.source_id for u in updates_to_process],
                    aggregation_weight=normalized_weights,
                    timestamp=time.time()
                )
                
                # Update statistics
                self.aggregation_stats['total_aggregations'] += 1
                self.aggregation_stats['successful_aggregations'] += 1
                self.aggregation_stats['avg_participants'] = (
                    (self.aggregation_stats['avg_participants'] * (self.aggregation_stats['total_aggregations'] - 1) + 
                     len(updates_to_process)) / self.aggregation_stats['total_aggregations']
                )
                
                # Clear processed updates
                self.aggregation_buffer.updates.clear()
                self.aggregation_buffer.weights.clear()
                self.aggregation_buffer.pending_count = 0
                self.aggregation_buffer.last_aggregation = time.time()
                
                # Broadcast new global model to vehicles
                self._broadcast_global_model(result)
        
        except Exception as e:
            print(f"Aggregation failed: {e}")
    
    def _aggregate_models(self, updates: List[ModelUpdate], weights: Dict[str, float]) -> Optional[Dict[str, torch.Tensor]]:
        """Perform weighted model aggregation"""
        if not updates:
            return None
        
        aggregated_model = {}
        
        # Get model structure from first update
        first_update_data = updates[0].update_data
        
        if isinstance(first_update_data, dict):
            # Dictionary-based model
            for key in first_update_data.keys():
                weighted_params = []
                
                for update in updates:
                    if isinstance(update.update_data, dict) and key in update.update_data:
                        weight = weights.get(update.source_id, 0.0)
                        if weight > 0:
                            weighted_params.append(update.update_data[key] * weight)
                
                if weighted_params:
                    aggregated_model[key] = torch.stack(weighted_params).sum(dim=0)
        
        elif isinstance(first_update_data, torch.Tensor):
            # Single tensor model
            weighted_params = []
            for update in updates:
                if isinstance(update.update_data, torch.Tensor):
                    weight = weights.get(update.source_id, 0.0)
                    if weight > 0:
                        weighted_params.append(update.update_data * weight)
            
            if weighted_params:
                aggregated_model = torch.stack(weighted_params).sum(dim=0)
        
        return aggregated_model
    
    def submit_error_signal(self, source_id: str, error_signal: torch.Tensor, 
                          propagation_targets: List[str]):
        """Submit error signal for lazy propagation"""
        error_data = ErrorPropagation(
            error_signals={source_id: error_signal},
            accumulation_threshold=ERROR_ACCUMULATION_THRESHOLD,
            propagation_targets=propagation_targets,
            propagation_count=0
        )
        
        self.error_propagation_queue.put(error_data)
    
    def _should_propagate_errors(self, error_data: ErrorPropagation) -> bool:
        """Determine if errors should be propagated"""
        # Check accumulation threshold
        total_error = sum(torch.norm(error).item() for error in error_data.error_signals.values())
        return total_error >= ERROR_ACCUMULATION_THRESHOLD
    
    def _propagate_errors(self, error_data: ErrorPropagation):
        """Propagate accumulated errors"""
        try:
            for target_id in error_data.propagation_targets:
                if target_id in error_data.error_signals:
                    # Accumulate error for target
                    if target_id in self.error_accumulator:
                        self.error_accumulator[target_id] += error_data.error_signals[target_id]
                    else:
                        self.error_accumulator[target_id] = error_data.error_signals[target_id].clone()
            
            # Increment propagation count
            error_data.propagation_count += 1
            
        except Exception as e:
            print(f"Error propagation failed: {e}")
    
    def _propagate_accumulated_errors(self):
        """Propagate all accumulated errors"""
        if not self.error_accumulator:
            return
        
        # Create consolidated error propagation
        error_data = ErrorPropagation(
            error_signals=dict(self.error_accumulator),
            accumulation_threshold=ERROR_ACCUMULATION_THRESHOLD,
            propagation_targets=list(self.error_accumulator.keys()),
            propagation_count=1
        )
        
        self._propagate_errors(error_data)
        
        # Clear accumulator
        self.error_accumulator.clear()
    
    def _broadcast_global_model(self, result: AggregationResult):
        """Broadcast new global model to vehicles"""
        # This would interface with the communication module
        # For now, just log the broadcast
        print(f"Broadcasting global model to {len(result.participating_sources)} vehicles")
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model"""
        return self.global_model.copy() if self.global_model else {}
    
    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get aggregation engine statistics"""
        return {
            **self.aggregation_stats,
            'buffer_size': len(self.aggregation_buffer.updates),
            'pending_updates': self.aggregation_buffer.pending_count,
            'last_aggregation': self.aggregation_buffer.last_aggregation,
            'error_accumulator_size': len(self.error_accumulator)
        }
    
    def shutdown(self):
        """Shutdown aggregation engine"""
        self.stop_event.set()
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5.0)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown()