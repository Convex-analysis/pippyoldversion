"""
Mobility Prediction Module using DTMC (Discrete-Time Markov Chain) Modeling

Predicts vehicle positions and movement patterns to optimize template matching
and pipeline formation decisions.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import time
import math
from dataclasses import dataclass

from ..core.types import VehicleInfo, MobilityPrediction
from ..core.constants import (
    DTMC_PREDICTION_HORIZON, 
    DTMC_TRANSITION_MEMORY,
    MOBILITY_UPDATE_INTERVAL,
    POSITION_TOLERANCE
)

@dataclass
class MobilityState:
    """Discrete mobility state for DTMC modeling"""
    region_id: int  # spatial region
    velocity_bin: int  # velocity category
    direction_bin: int  # direction category
    timestamp: float

class DTMCModel:
    """Discrete-Time Markov Chain model for mobility prediction"""
    
    def __init__(self, num_regions: int = 100, velocity_bins: int = 5, direction_bins: int = 8):
        self.num_regions = num_regions
        self.velocity_bins = velocity_bins
        self.direction_bins = direction_bins
        
        # Transition matrix: state -> next_state -> probability
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.state_counts = defaultdict(int)
        self.transition_counts = defaultdict(int)
        
        # State sequence history for each vehicle
        self.vehicle_histories = defaultdict(lambda: deque(maxlen=DTMC_TRANSITION_MEMORY))
        
        # Spatial regions grid
        self.grid_size = 1000  # meters per grid cell
        self.region_size = 100  # meters per region
        
        # Velocity bin boundaries
        self.velocity_boundaries = np.linspace(0, 50, velocity_bins + 1)  # 0-50 m/s
        
        # Direction bin boundaries (radians)
        self.direction_boundaries = np.linspace(0, 2 * np.pi, direction_bins + 1)
        
    def _discretize_position(self, position: Tuple[float, float]) -> int:
        """Convert continuous position to discrete region ID"""
        x, y = position
        region_x = int(x // self.region_size)
        region_y = int(y // self.region_size)
        return region_x * (self.grid_size // self.region_size) + region_y
    
    def _discretize_velocity(self, velocity: float) -> int:
        """Convert continuous velocity to discrete bin"""
        return np.digitize(velocity, self.velocity_boundaries) - 1
    
    def _discretize_direction(self, direction: float) -> int:
        """Convert continuous direction to discrete bin"""
        # Normalize to [0, 2π]
        direction = direction % (2 * np.pi)
        return np.digitize(direction, self.direction_boundaries) - 1
    
    def _create_state(self, vehicle_info: VehicleInfo) -> str:
        """Create discrete mobility state from vehicle info"""
        region = self._discretize_position(vehicle_info.position)
        velocity_bin = self._discretize_velocity(vehicle_info.velocity)
        direction_bin = self._discretize_direction(vehicle_info.direction)
        
        return f"{region}_{velocity_bin}_{direction_bin}"
    
    def update_state(self, vehicle_info: VehicleInfo):
        """Update vehicle state and transition matrix"""
        current_state = self._create_state(vehicle_info)
        current_time = vehicle_info.last_seen
        
        # Create mobility state object
        mobility_state = MobilityState(
            region_id=self._discretize_position(vehicle_info.position),
            velocity_bin=self._discretize_velocity(vehicle_info.velocity),
            direction_bin=self._discretize_direction(vehicle_info.direction),
            timestamp=current_time
        )
        
        # Update vehicle history
        history = self.vehicle_histories[vehicle_info.vehicle_id]
        if len(history) > 0:
            previous_state = history[-1]
            prev_state_str = self._create_state_from_mobility_state(previous_state)
            
            # Update transition counts
            self.transition_counts[(prev_state_str, current_state)] += 1
            self.state_counts[prev_state_str] += 1
        
        history.append(mobility_state)
    
    def _create_state_from_mobility_state(self, mobility_state: MobilityState) -> str:
        """Create state string from mobility state object"""
        return f"{mobility_state.region_id}_{mobility_state.velocity_bin}_{mobility_state.direction_bin}"
    
    def _normalize_transition_probabilities(self):
        """Normalize transition counts to probabilities"""
        for prev_state in self.transition_matrix:
            total = sum(self.transition_matrix[prev_state].values())
            if total > 0:
                for next_state in self.transition_matrix[prev_state]:
                    self.transition_matrix[prev_state][next_state] /= total
    
    def predict_next_states(self, vehicle_id: str, horizon: float = DTMC_PREDICTION_HORIZON) -> List[MobilityPrediction]:
        """Predict future states using DTMC"""
        if vehicle_id not in self.vehicle_histories:
            return []
        
        history = self.vehicle_histories[vehicle_id]
        if len(history) == 0:
            return []
        
        current_state = self._create_state_from_mobility_state(history[-1])
        
        # Normalize transition probabilities
        self._normalize_transition_probabilities()
        
        predictions = []
        current_state_str = current_state
        
        for step in range(int(horizon / MOBILITY_UPDATE_INTERVAL)):
            if current_state_str not in self.transition_matrix:
                break
            
            # Get next state probabilities
            next_states = self.transition_matrix[current_state_str]
            if not next_states:
                break
            
            # Find most probable next state
            next_state_str = max(next_states.items(), key=lambda x: x[1])
            next_state, probability = next_state_str
            
            # Convert state string back to mobility parameters
            region, vel_bin, dir_bin = map(int, next_state.split('_'))
            
            # Estimate position from region
            grid_cols = self.grid_size // self.region_size
            region_x = region // grid_cols
            region_y = region % grid_cols
            predicted_x = region_x * self.region_size + self.region_size / 2
            predicted_y = region_y * self.region_size + self.region_size / 2
            
            # Estimate velocity from bin
            if vel_bin < len(self.velocity_boundaries) - 1:
                predicted_velocity = (self.velocity_boundaries[vel_bin] + self.velocity_boundaries[vel_bin + 1]) / 2
            else:
                predicted_velocity = self.velocity_boundaries[-1]
            
            # Estimate direction from bin
            if dir_bin < len(self.direction_boundaries) - 1:
                predicted_direction = (self.direction_boundaries[dir_bin] + self.direction_boundaries[dir_bin + 1]) / 2
            else:
                predicted_direction = 0
            
            future_time = time.time() + (step + 1) * MOBILITY_UPDATE_INTERVAL
            
            prediction = MobilityPrediction(
                vehicle_id=vehicle_id,
                predicted_position=(predicted_x, predicted_y),
                predicted_time=future_time,
                confidence=probability,
                transition_probabilities=dict(next_states)
            )
            
            predictions.append(prediction)
            current_state_str = next_state_str
        
        return predictions
    
    def calculate_connection_probability(self, vehicle1_id: str, vehicle2_id: str, 
                                       future_time: float) -> float:
        """Calculate probability that two vehicles will be connected at future time"""
        pred1 = self.predict_next_states(vehicle1_id, future_time - time.time())
        pred2 = self.predict_next_states(vehicle2_id, future_time - time.time())
        
        if not pred1 or not pred2:
            return 0.0
        
        # Find predictions closest to future_time
        p1 = min(pred1, key=lambda x: abs(x.predicted_time - future_time))
        p2 = min(pred2, key=lambda x: abs(x.predicted_time - future_time))
        
        # Calculate distance
        distance = math.sqrt(
            (p1.predicted_position[0] - p2.predicted_position[0]) ** 2 +
            (p1.predicted_position[1] - p2.predicted_position[1]) ** 2
        )
        
        # Connection probability based on distance and prediction confidence
        if distance > 300:  # Maximum V2V range
            return 0.0
        
        distance_factor = 1.0 - (distance / 300.0)
        confidence_factor = (p1.confidence + p2.confidence) / 2.0
        
        return distance_factor * confidence_factor
    
    def get_trajectory_statistics(self, vehicle_id: str) -> Dict[str, float]:
        """Get trajectory statistics for a vehicle"""
        if vehicle_id not in self.vehicle_histories:
            return {}
        
        history = self.vehicle_histories[vehicle_id]
        if len(history) < 2:
            return {}
        
        positions = [(state.region_id * self.region_size, 0) for state in history]
        velocities = [self.velocity_boundaries[state.velocity_bin] if state.velocity_bin < len(self.velocity_boundaries) - 1 
                     else self.velocity_boundaries[-1] for state in history]
        
        # Calculate average velocity
        avg_velocity = np.mean(velocities)
        
        # Calculate direction changes
        direction_changes = 0
        for i in range(1, len(history)):
            if history[i].direction_bin != history[i-1].direction_bin:
                direction_changes += 1
        
        direction_change_rate = direction_changes / len(history)
        
        return {
            "avg_velocity": avg_velocity,
            "direction_change_rate": direction_change_rate,
            "trajectory_length": len(history),
            "prediction_confidence": np.mean([pred.confidence for pred in self.predict_next_states(vehicle_id)])
        }

class MobilityPredictor:
    """Main mobility prediction service"""
    
    def __init__(self):
        self.dtmc_model = DTMCModel()
        self.last_update_time = time.time()
        
    def update_vehicle_mobility(self, vehicle_info: VehicleInfo):
        """Update mobility data for a vehicle"""
        self.dtmc_model.update_state(vehicle_info)
        self.last_update_time = time.time()
    
    def predict_mobility(self, vehicle_id: str, horizon: float = DTMC_PREDICTION_HORIZON) -> List[MobilityPrediction]:
        """Predict mobility for a vehicle"""
        return self.dtmc_model.predict_next_states(vehicle_id, horizon)
    
    def predict_pipeline_stability(self, vehicle_ids: List[str], duration: float) -> float:
        """Predict pipeline stability probability over duration"""
        if len(vehicle_ids) < 2:
            return 0.0
        
        stability_prob = 1.0
        
        # Check pairwise connection probabilities
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                future_time = time.time() + duration
                conn_prob = self.dtmc_model.calculate_connection_probability(
                    vehicle_ids[i], vehicle_ids[j], future_time
                )
                stability_prob *= conn_prob
        
        return stability_prob
    
    def get_mobility_features(self, vehicle_id: str) -> Dict[str, float]:
        """Extract mobility features for template matching"""
        stats = self.dtmc_model.get_trajectory_statistics(vehicle_id)
        predictions = self.predict_mobility(vehicle_id)
        
        features = stats.copy()
        
        if predictions:
            # Add prediction-based features
            features.update({
                "prediction_horizon_confidence": np.mean([p.confidence for p in predictions]),
                "predicted_displacement": predictions[-1].confidence if predictions else 0.0,
                "mobility_regularity": 1.0 - stats.get("direction_change_rate", 0.0)
            })
        
        return features