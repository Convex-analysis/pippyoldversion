"""
Fairness Mechanism and Error Propagation System for FHDP

Implements frequency-based fairness mechanism and lazy error propagation
to ensure equitable participation and communication efficiency.
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import queue
import torch

from .types import (
    FairnessMetrics, ErrorPropagation, ModelUpdate, VehicleInfo,
    TrainingMode, ResourceClass
)
from .constants import (
    FAIRNESS_WINDOW_SIZE, FAIRNESS_DECAY_FACTOR, MIN_PARTICIPATION_INTERVAL,
    ERROR_ACCUMULATION_THRESHOLD, MAX_ERROR_PROPAGATION_DELAY,
    WEIGHT_DECAY_FACTOR
)

@dataclass
class ParticipationRecord:
    """Record of vehicle participation"""
    vehicle_id: str
    timestamp: float
    training_mode: TrainingMode
    success: bool
    contribution_score: float
    data_size: int
    computation_cost: float

@dataclass
class FairnessState:
    """Current fairness state for a vehicle"""
    participation_count: int = 0
    last_participation: float = 0.0
    historical_contributions: deque = field(default_factory=lambda: deque(maxlen=FAIRNESS_WINDOW_SIZE))
    priority_score: float = 1.0
    exclusion_list: Set[str] = field(default_factory=set)
    
class FrequencyBasedFairnessManager:
    """Manages frequency-based fairness for vehicle participation"""
    
    def __init__(self):
        self.fairness_states: Dict[str, FairnessState] = {}
        self.global_participation_history: List[ParticipationRecord] = []
        self.fairness_adjustments: Dict[str, float] = {}
        
        # Fairness parameters
        self.max_participation_rate = 0.1  # Maximum participations per second
        self.min_participation_interval = MIN_PARTICIPATION_INTERVAL
        self.fairness_weight_alpha = 0.2  # Weight for fairness adjustments
        
        # Statistical tracking
        self.fairness_stats = {
            'total_participations': 0,
            'excluded_vehicles': 0,
            'avg_priority_score': 0.0,
            'fairness_violations': 0
        }
        
        # Lock for thread safety
        self.fairness_lock = threading.Lock()
    
    def record_participation(self, vehicle_id: str, training_mode: TrainingMode,
                          success: bool, contribution_score: float,
                          data_size: int, computation_cost: float):
        """Record vehicle participation"""
        with self.fairness_lock:
            current_time = time.time()
            
            # Create participation record
            record = ParticipationRecord(
                vehicle_id=vehicle_id,
                timestamp=current_time,
                training_mode=training_mode,
                success=success,
                contribution_score=contribution_score,
                data_size=data_size,
                computation_cost=computation_cost
            )
            
            # Update global history
            self.global_participation_history.append(record)
            
            # Update vehicle fairness state
            if vehicle_id not in self.fairness_states:
                self.fairness_states[vehicle_id] = FairnessState()
            
            state = self.fairness_states[vehicle_id]
            state.participation_count += 1
            state.last_participation = current_time
            
            # Update historical contributions
            adjusted_score = contribution_score * (1.5 if training_mode == TrainingMode.PIPELINE else 1.0)
            if not success:
                adjusted_score *= 0.5  # Penalty for failure
            
            state.historical_contributions.append(adjusted_score)
            
            # Update priority score
            self._update_priority_score(vehicle_id)
            
            # Update statistics
            self.fairness_stats['total_participations'] += 1
            
            if not success:
                self.fairness_stats['fairness_violations'] += 1
    
    def _update_priority_score(self, vehicle_id: str):
        """Update priority score for fair participation"""
        state = self.fairness_states[vehicle_id]
        current_time = time.time()
        
        # Time-based factor (longer wait = higher priority)
        time_since_last = current_time - state.last_participation
        time_factor = min(3.0, 1.0 + time_since_last / self.min_participation_interval)
        
        # Historical contribution factor
        if state.historical_contributions:
            avg_contribution = np.mean(state.historical_contributions)
            contribution_factor = avg_contribution
        else:
            contribution_factor = 1.0
        
        # Participation frequency factor (penalize over-participation)
        recent_participations = [
            p for p in self.global_participation_history
            if p.vehicle_id == vehicle_id and current_time - p.timestamp < 300  # Last 5 minutes
        ]
        
        if len(recent_participations) > 0:
            time_span = max(1.0, recent_participations[-1].timestamp - recent_participations[0].timestamp)
            frequency = len(recent_participations) / time_span
            frequency_factor = max(0.1, 1.0 - frequency / self.max_participation_rate)
        else:
            frequency_factor = 1.0
        
        # Calculate final priority score
        new_priority = time_factor * contribution_factor * frequency_factor
        
        # Apply smoothing
        state.priority_score = (
            (1 - self.fairness_weight_alpha) * state.priority_score + 
            self.fairness_weight_alpha * new_priority
        )
        
        # Update global average
        all_scores = [s.priority_score for s in self.fairness_states.values()]
        self.fairness_stats['avg_priority_score'] = np.mean(all_scores)
    
    def can_participate(self, vehicle_id: str, current_time: Optional[float] = None) -> Tuple[bool, str]:
        """Determine if vehicle can participate based on fairness constraints"""
        if current_time is None:
            current_time = time.time()
        
        with self.fairness_lock:
            if vehicle_id not in self.fairness_states:
                return True, "New vehicle"
            
            state = self.fairness_states[vehicle_id]
            
            # Check if vehicle is excluded
            if vehicle_id in state.exclusion_list:
                return False, "Vehicle excluded due to fairness violation"
            
            # Check minimum interval
            time_since_last = current_time - state.last_participation
            if time_since_last < self.min_participation_interval:
                remaining_time = self.min_participation_interval - time_since_last
                return False, f"Minimum interval not met ({remaining_time:.1f}s remaining)"
            
            # Check participation rate
            recent_participations = [
                p for p in self.global_participation_history
                if p.vehicle_id == vehicle_id and current_time - p.timestamp < 60  # Last minute
            ]
            
            if len(recent_participations) > 0:
                frequency = len(recent_participations) / 60.0
                if frequency > self.max_participation_rate:
                    return False, f"Participation rate too high ({frequency:.3f}/s)"
            
            return True, "Can participate"
    
    def get_fairness_metrics(self, vehicle_id: str) -> FairnessMetrics:
        """Get fairness metrics for vehicle"""
        with self.fairness_lock:
            if vehicle_id not in self.fairness_states:
                return FairnessMetrics(
                    vehicle_id=vehicle_id,
                    participation_count=0,
                    last_participation=0.0,
                    contribution_score=1.0,
                    priority_weight=1.0
                )
            
            state = self.fairness_states[vehicle_id]
            
            contribution_score = (
                np.mean(state.historical_contributions) if state.historical_contributions else 1.0
            )
            
            return FairnessMetrics(
                vehicle_id=vehicle_id,
                participation_count=state.participation_count,
                last_participation=state.last_participation,
                contribution_score=contribution_score,
                priority_weight=state.priority_score
            )
    
    def exclude_vehicle(self, vehicle_id: str, reason: str, duration: float = 300.0):
        """Temporarily exclude vehicle from participation"""
        with self.fairness_lock:
            if vehicle_id not in self.fairness_states:
                self.fairness_states[vehicle_id] = FairnessState()
            
            self.fairness_states[vehicle_id].exclusion_list.add(reason)
            self.fairness_stats['excluded_vehicles'] += 1
            
            # Schedule removal of exclusion
            threading.Timer(duration, self._remove_exclusion, args=[vehicle_id, reason]).start()
    
    def _remove_exclusion(self, vehicle_id: str, reason: str):
        """Remove exclusion for vehicle"""
        with self.fairness_lock:
            if vehicle_id in self.fairness_states:
                self.fairness_states[vehicle_id].exclusion_list.discard(reason)
                if not self.fairness_states[vehicle_id].exclusion_list:
                    self.fairness_stats['excluded_vehicles'] = max(0, self.fairness_stats['excluded_vehicles'] - 1)
    
    def get_participation_candidates(self, available_vehicles: List[str], 
                                  max_candidates: int = 10) -> List[Tuple[str, float]]:
        """Get ranked list of candidates for participation"""
        with self.fairness_lock:
            candidates = []
            
            for vehicle_id in available_vehicles:
                can_participate, reason = self.can_participate(vehicle_id)
                if can_participate:
                    state = self.fairness_states.get(vehicle_id)
                    priority_score = state.priority_score if state else 1.0
                    candidates.append((vehicle_id, priority_score))
            
            # Sort by priority score (higher is better)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            return candidates[:max_candidates]
    
    def get_fairness_statistics(self) -> Dict[str, float]:
        """Get fairness system statistics"""
        with self.fairness_lock:
            current_time = time.time()
            
            # Calculate additional statistics
            total_vehicles = len(self.fairness_states)
            active_vehicles = len([
                s for s in self.fairness_states.values()
                if current_time - s.last_participation < 300  # Active in last 5 minutes
            ])
            
            # Calculate Gini coefficient for fairness
            priority_scores = [s.priority_score for s in self.fairness_states.values()]
            gini_coefficient = self._calculate_gini_coefficient(priority_scores)
            
            return {
                **self.fairness_stats,
                'total_vehicles': total_vehicles,
                'active_vehicles': active_vehicles,
                'gini_coefficient': gini_coefficient,
                'avg_participations_per_vehicle': (
                    self.fairness_stats['total_participations'] / max(1, total_vehicles)
                )
            }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for fairness measurement"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(values)
        cumulative_sum = 0
        for i, value in enumerate(sorted_values, 1):
            cumulative_sum += i * value
        
        sum_values = sum(sorted_values)
        if sum_values == 0:
            return 0.0
        
        gini = (2 * cumulative_sum) / (n * sum_values) - (n + 1) / n
        return gini

class LazyErrorPropagationManager:
    """Manages lazy error propagation for communication efficiency"""
    
    def __init__(self):
        self.error_accumulators: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.error_thresholds: Dict[str, float] = defaultdict(lambda: ERROR_ACCUMULATION_THRESHOLD)
        self.last_propagation: Dict[str, float] = {}
        
        # Propagation statistics
        self.propagation_stats = {
            'total_accumulations': 0,
            'total_propagations': 0,
            'avg_accumulation_size': 0.0,
            'total_error_reduced': 0.0,
            'communication_savings': 0.0
        }
        
        # Propagation queue
        self.propagation_queue = queue.Queue()
        self.propagation_worker_thread = None
        self.propagation_active = False
        
        # Compression parameters
        self.compression_enabled = True
        self.compression_ratio = 0.7
        self.max_delay = MAX_ERROR_PROPAGATION_DELAY
    
    def start_propagation_service(self):
        """Start error propagation service"""
        if self.propagation_active:
            return
        
        self.propagation_active = True
        self.propagation_worker_thread = threading.Thread(
            target=self._propagation_worker, daemon=True
        )
        self.propagation_worker_thread.start()
    
    def stop_propagation_service(self):
        """Stop error propagation service"""
        self.propagation_active = False
        
        if self.propagation_worker_thread:
            self.propagation_worker_thread.join(timeout=5.0)
    
    def accumulate_error(self, source_id: str, error_signal: torch.Tensor, 
                        target_ids: List[str], custom_threshold: Optional[float] = None):
        """Accumulate error signal for lazy propagation"""
        threshold = custom_threshold or self.error_thresholds[source_id]
        
        # Add to accumulator
        self.error_accumulators[source_id].append(error_signal)
        
        # Update statistics
        self.propagation_stats['total_accumulations'] += 1
        
        # Check if propagation should be triggered
        if self._should_propagate_error(source_id, threshold):
            self._queue_propagation(source_id, target_ids)
    
    def _should_propagate_error(self, source_id: str, threshold: float) -> bool:
        """Determine if accumulated error should be propagated"""
        current_time = time.time()
        
        # Check time threshold
        time_since_last = current_time - self.last_propagation.get(source_id, 0)
        if time_since_last >= self.max_delay:
            return True
        
        # Check accumulation threshold
        accumulated_errors = self.error_accumulators[source_id]
        if not accumulated_errors:
            return False
        
        # Calculate total error magnitude
        total_error = sum(torch.norm(error).item() for error in accumulated_errors)
        
        return total_error >= threshold
    
    def _queue_propagation(self, source_id: str, target_ids: List[str]):
        """Queue error for propagation"""
        error_data = ErrorPropagation(
            error_signals={source_id: accumulated for accumulated in self.error_accumulators[source_id]},
            accumulation_threshold=self.error_thresholds[source_id],
            propagation_targets=target_ids,
            propagation_count=len(self.error_accumulators[source_id])
        )
        
        self.propagation_queue.put((source_id, error_data))
        
        # Clear accumulator
        self.error_accumulators[source_id].clear()
        self.last_propagation[source_id] = time.time()
        
        # Update statistics
        self.propagation_stats['total_propagations'] += 1
    
    def _propagation_worker(self):
        """Error propagation worker thread"""
        while self.propagation_active:
            try:
                # Get propagation task
                source_id, error_data = self.propagation_queue.get(timeout=1.0)
                
                # Process propagation
                self._process_error_propagation(source_id, error_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error propagation worker error: {e}")
    
    def _process_error_propagation(self, source_id: str, error_data: ErrorPropagation):
        """Process error propagation"""
        try:
            # Compress error signals if enabled
            if self.compression_enabled:
                compressed_errors = self._compress_error_signals(error_data.error_signals)
            else:
                compressed_errors = error_data.error_signals
            
            # Calculate propagation savings
            original_size = sum(
                torch.norm(error).numel() for error in error_data.error_signals.values()
            )
            compressed_size = sum(
                torch.norm(error).numel() for error in compressed_errors.values()
            ) if isinstance(compressed_errors, dict) else compressed_size
            
            savings = (original_size - compressed_size) / max(1, original_size)
            self.propagation_stats['communication_savings'] += savings
            
            # Update statistics
            avg_error_size = np.mean([
                torch.norm(error).item() for error in error_data.error_signals.values()
            ]) if error_data.error_signals else 0.0
            
            self.propagation_stats['avg_accumulation_size'] = (
                (self.propagation_stats['avg_accumulation_size'] * (self.propagation_stats['total_propagations'] - 1) + 
                 avg_error_size) / self.propagation_stats['total_propagations']
            )
            
            # In real implementation, this would send compressed_errors to targets
            for target_id in error_data.propagation_targets:
                self._send_error_signal(target_id, compressed_errors)
            
        except Exception as e:
            print(f"Error in processing propagation: {e}")
    
    def _compress_error_signals(self, error_signals: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress error signals for efficient transmission"""
        compressed = {}
        
        for source_id, error_signal in error_signals.items():
            # Simple compression: keep only top-k largest magnitude elements
            error_flat = error_signal.flatten()
            
            # Determine number of elements to keep based on compression ratio
            k = max(1, int(len(error_flat) * self.compression_ratio))
            
            # Get indices of top-k elements
            _, top_indices = torch.topk(torch.abs(error_flat), k)
            
            # Create sparse representation
            compressed_signal = torch.zeros_like(error_flat)
            compressed_signal[top_indices] = error_flat[top_indices]
            
            compressed[source_id] = compressed_signal.reshape(error_signal.shape)
        
        return compressed
    
    def _send_error_signal(self, target_id: str, error_data: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """Send error signal to target vehicle (placeholder)"""
        # In real implementation, this would use the communication layer
        pass
    
    def force_propagation(self, source_id: str, target_ids: List[str]):
        """Force immediate propagation of accumulated errors"""
        if self.error_accumulators[source_id]:
            self._queue_propagation(source_id, target_ids)
    
    def get_propagation_statistics(self) -> Dict[str, float]:
        """Get error propagation statistics"""
        total_accumulations = self.propagation_stats['total_accumulations']
        
        propagation_efficiency = (
            self.propagation_stats['total_propagations'] / max(1, total_accumulations)
        ) * 100
        
        avg_communication_savings = (
            self.propagation_stats['communication_savings'] / max(1, self.propagation_stats['total_propagations'])
        ) * 100
        
        return {
            **self.propagation_stats,
            'propagation_efficiency_percent': propagation_efficiency,
            'avg_communication_savings_percent': avg_communication_savings,
            'active_accumulators': len(self.error_accumulators),
            'total_pending_errors': sum(
                len(errors) for errors in self.error_accumulators.values()
            )
        }
    
    def set_custom_threshold(self, source_id: str, threshold: float):
        """Set custom error accumulation threshold for source"""
        self.error_thresholds[source_id] = threshold
    
    def clear_accumulator(self, source_id: str):
        """Clear error accumulator for source"""
        if source_id in self.error_accumulators:
            self.error_accumulators[source_id].clear()

class FairnessErrorManager:
    """Combined fairness and error propagation manager"""
    
    def __init__(self):
        self.fairness_manager = FrequencyBasedFairnessManager()
        self.error_manager = LazyErrorPropagationManager()
        
        # Integrated statistics
        self.integrated_stats = {
            'total_operations': 0,
            'fair_adjustments': 0,
            'error_propagations': 0
        }
    
    def start_services(self):
        """Start all services"""
        self.error_manager.start_propagation_service()
    
    def stop_services(self):
        """Stop all services"""
        self.error_manager.stop_propagation_service()
    
    def process_training_result(self, vehicle_id: str, training_mode: TrainingMode,
                              success: bool, contribution_score: float,
                              data_size: int, computation_cost: float,
                              error_signal: Optional[torch.Tensor] = None,
                              error_targets: Optional[List[str]] = None):
        """Process training result with fairness and error handling"""
        # Record participation for fairness
        self.fairness_manager.record_participation(
            vehicle_id, training_mode, success, contribution_score, data_size, computation_cost
        )
        
        # Accumulate error signal if provided
        if error_signal is not None and error_targets:
            self.error_manager.accumulate_error(vehicle_id, error_signal, error_targets)
        
        # Update integrated statistics
        self.integrated_stats['total_operations'] += 1
        
        if training_mode == TrainingMode.PIPELINE or contribution_score < 0.5:
            self.integrated_stats['fair_adjustments'] += 1
        
        if error_signal is not None:
            self.integrated_stats['error_propagations'] += 1
    
    def can_participate(self, vehicle_id: str) -> Tuple[bool, str]:
        """Check if vehicle can participate based on fairness"""
        return self.fairness_manager.can_participate(vehicle_id)
    
    def get_participation_candidates(self, available_vehicles: List[str],
                                  max_candidates: int = 10) -> List[Tuple[str, float]]:
        """Get ranked participation candidates"""
        return self.fairness_manager.get_participation_candidates(available_vehicles, max_candidates)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from both systems"""
        return {
            'integrated_stats': self.integrated_stats,
            'fairness_stats': self.fairness_manager.get_fairness_statistics(),
            'error_propagation_stats': self.error_manager.get_propagation_statistics(),
            'system_efficiency': {
                'fairness_coverage': (
                    self.integrated_stats['fair_adjustments'] / max(1, self.integrated_stats['total_operations'])
                ) * 100,
                'error_handling_coverage': (
                    self.integrated_stats['error_propagations'] / max(1, self.integrated_stats['total_operations'])
                ) * 100
            }
        }