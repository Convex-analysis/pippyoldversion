"""
Autonomous driving evaluation metrics for EVO-1 model

This module implements comprehensive evaluation metrics specifically
designed for autonomous driving performance assessment.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import cv2
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config import EvaluationConfig, ModelConfig


@dataclass
class TrajectoryEvaluation:
    """Trajectory evaluation results"""
    ade: float  # Average Displacement Error
    fde: float  # Final Displacement Error  
    miss_rate: float  # Miss rate (error > threshold)
    minade: float  # Minimum ADE across multiple predictions
    maxade: float  # Maximum ADE across multiple predictions
    trajectory_length_error: float  # Error in trajectory length
    heading_error: float  # Average heading error


@dataclass
class ControlEvaluation:
    """Control performance evaluation"""
    steering_mae: float  # Mean absolute error
    steering_rmse: float  # Root mean square error
    throttle_mae: float
    throttle_rmse: float
    brake_mae: float
    brake_rmse: float
    control_smoothness: float  # Jerk/smoothness metric
    control_frequency_error: float  # Deviation from target frequency


@dataclass
class SafetyEvaluation:
    """Safety-related evaluation metrics"""
    collision_rate: float  # Rate of collisions
    offroad_rate: float  # Rate of going off-road
    traffic_violation_rate: float  # Rate of traffic violations
    safety_margin_violations: float  # Violations of safety margins
    emergency_brake_rate: float  # Rate of emergency braking
    comfort_score: float  # Passenger comfort score


@dataclass
class EfficiencyEvaluation:
    """Efficiency evaluation metrics"""
    trip_time_error: float  # Error in trip time
    fuel_efficiency_score: float  # Fuel efficiency score
    average_speed_error: float  # Error in average speed
    path_efficiency: float  # Efficiency of path planning
    stop_count_error: float  # Error in number of stops


class DrivingMetricsEvaluator:
    """Comprehensive evaluator for autonomous driving metrics"""
    
    def __init__(
        self,
        config: EvaluationConfig,
        model_config: ModelConfig,
        output_dir: Optional[str] = None
    ):
        self.config = config
        self.model_config = model_config
        self.output_dir = output_dir
        
        # Setup output directory
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Evaluation thresholds
        self.ade_threshold = 2.0  # meters
        self.fde_threshold = 2.0  # meters
        self.safety_margin = config.safety_margin_m
        self.time_horizon = config.time_horizon_s
        
        # Initialize metrics storage
        self.trajectory_evaluations = []
        self.control_evaluations = []
        self.safety_evaluations = []
        self.efficiency_evaluations = []
        
        # Setup visualization
        self.setup_visualization()
    
    def setup_visualization(self):
        """Setup visualization for evaluation results"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def evaluate_trajectories(
        self,
        predicted_trajectories: torch.Tensor,
        ground_truth_trajectories: torch.Tensor,
        predicted_controls: Optional[torch.Tensor] = None
    ) -> TrajectoryEvaluation:
        """
        Evaluate trajectory prediction accuracy
        
        Args:
            predicted_trajectories: [B, T, 3] predicted waypoints
            ground_truth_trajectories: [B, T, 3] ground truth waypoints
            predicted_controls: [B, T, 3] control signals
        
        Returns:
            TrajectoryEvaluation
        """
        B, T, _ = predicted_trajectories.shape
        
        # Average Displacement Error (ADE)
        ade_errors = []
        fde_errors = []
        
        for i in range(B):
            pred_traj = predicted_trajectories[i].cpu().numpy()
            gt_traj = ground_truth_trajectories[i].cpu().numpy()
            
            # ADE - average error over all time steps
            ade_error = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1))
            ade_errors.append(ade_error)
            
            # FDE - error at final time step
            fde_error = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
            fde_errors.append(fde_error)
        
        ade = np.mean(ade_errors)
        fde = np.mean(fde_errors)
        
        # Miss rate (percentage with error > threshold)
        miss_rate = np.mean(np.array(ade_errors) > self.ade_threshold) * 100
        
        # Min and Max ADE
        minade = np.min(ade_errors)
        maxade = np.max(ade_errors)
        
        # Trajectory length error
        pred_lengths = self._compute_trajectory_lengths(predicted_trajectories)
        gt_lengths = self._compute_trajectory_lengths(ground_truth_trajectories)
        trajectory_length_error = np.mean(np.abs(pred_lengths.cpu().numpy() - gt_lengths.cpu().numpy()))
        
        # Heading error
        if predicted_controls is not None:
            heading_errors = []
            for i in range(B):
                pred_heading = self._compute_heading_from_trajectory(predicted_trajectories[i])
                gt_heading = self._compute_heading_from_trajectory(ground_truth_trajectories[i])
                
                heading_error = np.mean(np.abs(pred_heading - gt_heading))
                heading_errors.append(heading_error)
            
            heading_error = np.mean(heading_errors)
        else:
            heading_error = 0.0
        
        evaluation = TrajectoryEvaluation(
            ade=ade,
            fde=fde,
            miss_rate=miss_rate,
            minade=minade,
            maxade=maxade,
            trajectory_length_error=trajectory_length_error,
            heading_error=heading_error
        )
        
        self.trajectory_evaluations.append(evaluation)
        return evaluation
    
    def evaluate_controls(
        self,
        predicted_controls: torch.Tensor,
        ground_truth_controls: torch.Tensor
    ) -> ControlEvaluation:
        """
        Evaluate control signal accuracy
        
        Args:
            predicted_controls: [B, T, 3] [steering, throttle, brake]
            ground_truth_controls: [B, T, 3] ground truth controls
        
        Returns:
            ControlEvaluation
        """
        B, T, _ = predicted_controls.shape
        
        # Extract individual control components
        pred_steering = predicted_controls[:, :, 0].cpu().numpy()
        gt_steering = ground_truth_controls[:, :, 0].cpu().numpy()
        
        pred_throttle = predicted_controls[:, :, 1].cpu().numpy()
        gt_throttle = ground_truth_controls[:, :, 1].cpu().numpy()
        
        pred_brake = predicted_controls[:, :, 2].cpu().numpy()
        gt_brake = ground_truth_controls[:, :, 2].cpu().numpy()
        
        # Compute metrics for each control type
        steering_mae = np.mean(np.abs(pred_steering - gt_steering))
        steering_rmse = np.sqrt(np.mean((pred_steering - gt_steering) ** 2))
        
        throttle_mae = np.mean(np.abs(pred_throttle - gt_throttle))
        throttle_rmse = np.sqrt(np.mean((pred_throttle - gt_throttle) ** 2))
        
        brake_mae = np.mean(np.abs(pred_brake - gt_brake))
        brake_rmse = np.sqrt(np.mean((pred_brake - gt_brake) ** 2))
        
        # Control smoothness (jerk)
        control_smoothness = self._compute_control_smoothness(predicted_controls)
        
        # Control frequency error
        control_frequency_error = self._compute_control_frequency_error(predicted_controls)
        
        evaluation = ControlEvaluation(
            steering_mae=steering_mae,
            steering_rmse=steering_rmse,
            throttle_mae=throttle_mae,
            throttle_rmse=throttle_rmse,
            brake_mae=brake_mae,
            brake_rmse=brake_rmse,
            control_smoothness=control_smoothness,
            control_frequency_error=control_frequency_error
        )
        
        self.control_evaluations.append(evaluation)
        return evaluation
    
    def evaluate_safety(
        self,
        predicted_trajectories: torch.Tensor,
        predicted_controls: torch.Tensor,
        environment_info: Optional[Dict[str, Any]] = None
    ) -> SafetyEvaluation:
        """
        Evaluate safety-related metrics
        
        Args:
            predicted_trajectories: [B, T, 3] predicted waypoints
            predicted_controls: [B, T, 3] control signals
            environment_info: Additional environment information
        
        Returns:
            SafetyEvaluation
        """
        B, T, _ = predicted_trajectories.shape
        
        collision_count = 0
        offroad_count = 0
        traffic_violation_count = 0
        safety_margin_violations = 0
        emergency_brake_count = 0
        
        comfort_scores = []
        
        for i in range(B):
            traj = predicted_trajectories[i].cpu().numpy()
            controls = predicted_controls[i].cpu().numpy()
            
            # Check collisions (simplified - requires environment info)
            if environment_info and 'obstacles' in environment_info:
                collision = self._check_collisions(traj, environment_info['obstacles'])
                if collision:
                    collision_count += 1
            else:
                # Simplified collision check based on trajectory validity
                if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                    collision_count += 1
            
            # Check offroad (simplified - check if y-coordinate is too large)
            offroad = np.any(np.abs(traj[:, 1]) > 5.0)  # 5m lateral threshold
            if offroad:
                offroad_count += 1
            
            # Check traffic violations (simplified - excessive speed)
            max_speed = np.max(np.linalg.norm(np.diff(traj, axis=0), axis=1)) * 10  # Convert to m/s
            if max_speed > 35.0:  # 35 m/s (126 km/h) speed limit
                traffic_violation_count += 1
            
            # Check safety margin violations
            if environment_info and 'other_vehicles' in environment_info:
                margin_violations = self._check_safety_margins(
                    traj, environment_info['other_vehicles']
                )
                safety_margin_violations += margin_violations
            
            # Check emergency braking
            brake_values = controls[:, 2]
            emergency_brake = np.any(brake_values > 0.8)  # 80% brake threshold
            if emergency_brake:
                emergency_brake_count += 1
            
            # Comfort score (based on acceleration and jerk)
            comfort_score = self._compute_comfort_score(traj, controls)
            comfort_scores.append(comfort_score)
        
        # Compute rates
        collision_rate = collision_count / B * 100
        offroad_rate = offroad_count / B * 100
        traffic_violation_rate = traffic_violation_count / B * 100
        emergency_brake_rate = emergency_brake_count / B * 100
        
        # Average comfort score
        comfort_score = np.mean(comfort_scores) if comfort_scores else 0.0
        
        evaluation = SafetyEvaluation(
            collision_rate=collision_rate,
            offroad_rate=offroad_rate,
            traffic_violation_rate=traffic_violation_rate,
            safety_margin_violations=safety_margin_violations,
            emergency_brake_rate=emergency_brake_rate,
            comfort_score=comfort_score
        )
        
        self.safety_evaluations.append(evaluation)
        return evaluation
    
    def evaluate_efficiency(
        self,
        predicted_trajectories: torch.Tensor,
        predicted_controls: torch.Tensor,
        reference_trajectories: Optional[torch.Tensor] = None
    ) -> EfficiencyEvaluation:
        """
        Evaluate efficiency metrics
        
        Args:
            predicted_trajectories: [B, T, 3] predicted waypoints
            predicted_controls: [B, T, 3] control signals
            reference_trajectories: [B, T, 3] reference trajectories
        
        Returns:
            EfficiencyEvaluation
        """
        B, T, _ = predicted_trajectories.shape
        
        trip_time_errors = []
        fuel_efficiency_scores = []
        average_speed_errors = []
        path_efficiency_scores = []
        stop_count_errors = []
        
        for i in range(B):
            traj = predicted_trajectories[i].cpu().numpy()
            controls = predicted_controls[i].cpu().numpy()
            
            # Trip time error (relative to reference)
            if reference_trajectories is not None:
                ref_traj = reference_trajectories[i].cpu().numpy()
                pred_distance = self._compute_trajectory_length(traj)
                ref_distance = self._compute_trajectory_length(ref_traj)
                
                # Estimate trip time based on average speed
                pred_avg_speed = np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1)) * 10
                ref_avg_speed = np.mean(np.linalg.norm(np.diff(ref_traj, axis=0), axis=1)) * 10
                
                if ref_avg_speed > 0:
                    pred_time = pred_distance / max(pred_avg_speed, 1.0)
                    ref_time = ref_distance / max(ref_avg_speed, 1.0)
                    trip_time_error = abs(pred_time - ref_time) / ref_time * 100
                else:
                    trip_time_error = 0.0
            else:
                trip_time_error = 0.0
            
            trip_time_errors.append(trip_time_error)
            
            # Fuel efficiency score (based on smooth acceleration and minimal braking)
            acceleration = np.diff(controls[:, :2], axis=0)  # Changes in steering and throttle
            fuel_score = 100.0 / (1.0 + np.mean(np.abs(acceleration)) * 10 + np.mean(controls[:, 2]) * 5)
            fuel_efficiency_scores.append(min(fuel_score, 100.0))
            
            # Average speed error
            avg_speed = np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1)) * 10
            target_speed = 15.0  # 15 m/s target speed
            average_speed_errors.append(abs(avg_speed - target_speed) / target_speed * 100)
            
            # Path efficiency (directness)
            if len(traj) > 1:
                direct_distance = np.linalg.norm(traj[-1] - traj[0])
                actual_distance = self._compute_trajectory_length(traj)
                path_efficiency = direct_distance / max(actual_distance, 1.0) * 100
            else:
                path_efficiency = 100.0
            
            path_efficiency_scores.append(min(path_efficiency, 100.0))
            
            # Stop count error (number of braking events)
            brake_events = np.sum(controls[:, 2] > 0.5)
            target_stops = 1  # Assume 1 stop per trajectory
            stop_count_errors.append(abs(brake_events - target_stops))
        
        evaluation = EfficiencyEvaluation(
            trip_time_error=np.mean(trip_time_errors),
            fuel_efficiency_score=np.mean(fuel_efficiency_scores),
            average_speed_error=np.mean(average_speed_errors),
            path_efficiency=np.mean(path_efficiency_scores),
            stop_count_error=np.mean(stop_count_errors)
        )
        
        self.efficiency_evaluations.append(evaluation)
        return evaluation
    
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        model_output: Dict[str, torch.Tensor],
        environment_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a complete batch of predictions
        
        Args:
            batch: Ground truth data
            model_output: Model predictions
            environment_info: Additional environment information
        
        Returns:
            Combined evaluation results
        """
        results = {}
        
        # Extract predictions and targets
        if 'waypoints' in model_output:
            predicted_trajectories = model_output['waypoints']
        else:
            predicted_trajectories = self._controls_to_waypoints(model_output['controls'])
        
        ground_truth_trajectories = batch['future_controls']  # Convert controls to waypoints
        ground_truth_trajectories = self._controls_to_waypoints(ground_truth_trajectories)
        
        predicted_controls = model_output['controls']
        ground_truth_controls = batch['future_controls']
        
        # Evaluate trajectories
        if self.config.evaluate_ade:
            trajectory_eval = self.evaluate_trajectories(
                predicted_trajectories,
                ground_truth_trajectories,
                predicted_controls
            )
            results['trajectory'] = trajectory_eval
        
        # Evaluate controls
        if self.config.evaluate_l2_error:
            control_eval = self.evaluate_controls(
                predicted_controls,
                ground_truth_controls
            )
            results['control'] = control_eval
        
        # Evaluate safety
        if self.config.evaluate_collision_rate:
            safety_eval = self.evaluate_safety(
                predicted_trajectories,
                predicted_controls,
                environment_info
            )
            results['safety'] = safety_eval
        
        # Evaluate efficiency
        if self.config.evaluate_offroad_rate:  # Using this flag as proxy for efficiency
            efficiency_eval = self.evaluate_efficiency(
                predicted_trajectories,
                predicted_controls
            )
            results['efficiency'] = efficiency_eval
        
        return results
    
    def compute_overall_score(self, evaluations: Dict[str, Any]) -> float:
        """Compute overall driving performance score"""
        score = 0.0
        total_weight = 0.0
        
        # Trajectory accuracy (weight: 0.3)
        if 'trajectory' in evaluations:
            traj_eval = evaluations['trajectory']
            traj_score = 100.0 - traj_eval.ade * 10 - self.ade_threshold * 5
            traj_score = max(0.0, min(100.0, traj_score))
            score += traj_score * 0.3
            total_weight += 0.3
        
        # Control accuracy (weight: 0.2)
        if 'control' in evaluations:
            ctrl_eval = evaluations['control']
            ctrl_score = 100.0 - ctrl_eval.steering_rmse * 50 - ctrl_eval.throttle_rmse * 50
            ctrl_score = max(0.0, min(100.0, ctrl_score))
            score += ctrl_score * 0.2
            total_weight += 0.2
        
        # Safety (weight: 0.3)
        if 'safety' in evaluations:
            safety_eval = evaluations['safety']
            safety_score = 100.0 - safety_eval.collision_rate - safety_eval.offroad_rate - safety_eval.emergency_brake_rate
            safety_score += safety_eval.comfort_score * 0.3
            safety_score = max(0.0, min(100.0, safety_score))
            score += safety_score * 0.3
            total_weight += 0.3
        
        # Efficiency (weight: 0.2)
        if 'efficiency' in evaluations:
            eff_eval = evaluations['efficiency']
            eff_score = (eff_eval.fuel_efficiency_score + eff_eval.path_efficiency) / 2
            eff_score = max(0.0, min(100.0, eff_score))
            score += eff_score * 0.2
            total_weight += 0.2
        
        return score / max(total_weight, 0.1) if total_weight > 0 else 0.0
    
    def visualize_results(self, save_path: Optional[str] = None) -> None:
        """Visualize evaluation results"""
        if not self.trajectory_evaluations:
            logging.warning("No trajectory evaluations to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Autonomous Driving Evaluation Results', fontsize=16)
        
        # ADE and FDE distribution
        ade_values = [eval.ade for eval in self.trajectory_evaluations]
        fde_values = [eval.fde for eval in self.trajectory_evaluations]
        
        axes[0, 0].hist(ade_values, bins=20, alpha=0.7, label='ADE')
        axes[0, 0].hist(fde_values, bins=20, alpha=0.7, label='FDE')
        axes[0, 0].set_title('Trajectory Error Distribution')
        axes[0, 0].set_xlabel('Error (meters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Control errors
        if self.control_evaluations:
            steering_errors = [eval.steering_rmse for eval in self.control_evaluations]
            throttle_errors = [eval.throttle_rmse for eval in self.control_evaluations]
            brake_errors = [eval.brake_rmse for eval in self.control_evaluations]
            
            axes[0, 1].boxplot([steering_errors, throttle_errors, brake_errors],
                              labels=['Steering', 'Throttle', 'Brake'])
            axes[0, 1].set_title('Control RMSE Distribution')
            axes[0, 1].set_ylabel('RMSE')
        
        # Safety metrics
        if self.safety_evaluations:
            collision_rates = [eval.collision_rate for eval in self.safety_evaluations]
            offroad_rates = [eval.offroad_rate for eval in self.safety_evaluations]
            comfort_scores = [eval.comfort_score for eval in self.safety_evaluations]
            
            axes[0, 2].scatter(collision_rates, comfort_scores, alpha=0.6)
            axes[0, 2].set_xlabel('Collision Rate (%)')
            axes[0, 2].set_ylabel('Comfort Score')
            axes[0, 2].set_title('Safety vs Comfort')
        
        # Efficiency metrics
        if self.efficiency_evaluations:
            fuel_scores = [eval.fuel_efficiency_score for eval in self.efficiency_evaluations]
            path_eff_scores = [eval.path_efficiency for eval in self.efficiency_evaluations]
            
            axes[1, 0].scatter(fuel_scores, path_eff_scores, alpha=0.6)
            axes[1, 0].set_xlabel('Fuel Efficiency Score')
            axes[1, 0].set_ylabel('Path Efficiency')
            axes[1, 0].set_title('Efficiency Metrics')
        
        # Overall score distribution
        if self.trajectory_evaluations and self.control_evaluations:
            overall_scores = []
            for i in range(len(self.trajectory_evaluations)):
                evaluations = {
                    'trajectory': self.trajectory_evaluations[i],
                    'control': self.control_evaluations[i] if i < len(self.control_evaluations) else None,
                    'safety': self.safety_evaluations[i] if i < len(self.safety_evaluations) else None,
                    'efficiency': self.efficiency_evaluations[i] if i < len(self.efficiency_evaluations) else None
                }
                score = self.compute_overall_score(evaluations)
                overall_scores.append(score)
            
            axes[1, 1].hist(overall_scores, bins=20, alpha=0.7, color='green')
            axes[1, 1].set_title('Overall Performance Score')
            axes[1, 1].set_xlabel('Score')
            axes[1, 1].set_ylabel('Frequency')
        
        # Combined metrics radar chart
        if self.trajectory_evaluations and self.control_evaluations:
            avg_ade = np.mean([eval.ade for eval in self.trajectory_evaluations])
            avg_steering = np.mean([eval.steering_rmse for eval in self.control_evaluations])
            avg_safety = np.mean([100.0 - eval.collision_rate for eval in self.safety_evaluations]) if self.safety_evaluations else 50.0
            avg_efficiency = np.mean([eval.fuel_efficiency_score for eval in self.efficiency_evaluations]) if self.efficiency_evaluations else 50.0
            
            # Normalize to 0-100 scale
            metrics = [
                max(0, 100 - avg_ade * 20),  # Trajectory (20 points per meter)
                max(0, 100 - avg_steering * 100),  # Control
                avg_safety,  # Safety
                avg_efficiency  # Efficiency
            ]
            labels = ['Trajectory', 'Control', 'Safety', 'Efficiency']
            
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            metrics += metrics[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            labels += labels[:1]
            
            ax = plt.subplot(2, 3, 6, projection='polar')
            ax.plot(angles, metrics, 'o-', linewidth=2)
            ax.fill(angles, metrics, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels[:-1])
            ax.set_ylim(0, 100)
            ax.set_title('Performance Radar Chart')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to file"""
        results = {
            'config': self.config.__dict__,
            'trajectory_evaluations': [
                {
                    'ade': eval.ade,
                    'fde': eval.fde,
                    'miss_rate': eval.miss_rate,
                    'minade': eval.minade,
                    'maxade': eval.maxade,
                    'trajectory_length_error': eval.trajectory_length_error,
                    'heading_error': eval.heading_error
                } for eval in self.trajectory_evaluations
            ],
            'control_evaluations': [
                {
                    'steering_mae': eval.steering_mae,
                    'steering_rmse': eval.steering_rmse,
                    'throttle_mae': eval.throttle_mae,
                    'throttle_rmse': eval.throttle_rmse,
                    'brake_mae': eval.brake_mae,
                    'brake_rmse': eval.brake_rmse,
                    'control_smoothness': eval.control_smoothness,
                    'control_frequency_error': eval.control_frequency_error
                } for eval in self.control_evaluations
            ],
            'safety_evaluations': [
                {
                    'collision_rate': eval.collision_rate,
                    'offroad_rate': eval.offroad_rate,
                    'traffic_violation_rate': eval.traffic_violation_rate,
                    'safety_margin_violations': eval.safety_margin_violations,
                    'emergency_brake_rate': eval.emergency_brake_rate,
                    'comfort_score': eval.comfort_score
                } for eval in self.safety_evaluations
            ],
            'efficiency_evaluations': [
                {
                    'trip_time_error': eval.trip_time_error,
                    'fuel_efficiency_score': eval.fuel_efficiency_score,
                    'average_speed_error': eval.average_speed_error,
                    'path_efficiency': eval.path_efficiency,
                    'stop_count_error': eval.stop_count_error
                } for eval in self.efficiency_evaluations
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Saved evaluation results to {filepath}")
    
    # Helper methods
    def _compute_trajectory_lengths(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute lengths of trajectories"""
        B, T, _ = trajectories.shape
        
        lengths = []
        for i in range(B):
            traj = trajectories[i]
            diffs = torch.diff(traj, dim=0)
            segment_lengths = torch.norm(diffs, dim=1)
            total_length = torch.sum(segment_lengths)
            lengths.append(total_length)
        
        return torch.stack(lengths)
    
    def _compute_trajectory_length(self, trajectory: np.ndarray) -> float:
        """Compute length of a single trajectory"""
        if len(trajectory) < 2:
            return 0.0
        
        diffs = np.diff(trajectory, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(segment_lengths)
    
    def _compute_heading_from_trajectory(self, trajectory: torch.Tensor) -> np.ndarray:
        """Compute heading angles from trajectory"""
        traj = trajectory.cpu().numpy()
        
        if len(traj) < 2:
            return np.array([0.0])
        
        diffs = np.diff(traj, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        return headings
    
    def _compute_control_smoothness(self, controls: torch.Tensor) -> float:
        """Compute control smoothness (inverse of jerk)"""
        # Compute second derivative (jerk)
        first_diff = torch.diff(controls, dim=1)
        second_diff = torch.diff(first_diff, dim=1)
        
        if second_diff.numel() == 0:
            return 0.0
        
        jerk = torch.mean(torch.norm(second_diff, dim=2))
        smoothness = 1.0 / (1.0 + jerk.item())
        
        return smoothness
    
    def _compute_control_frequency_error(self, controls: torch.Tensor) -> float:
        """Compute deviation from target control frequency"""
        target_freq = 10.0  # Hz
        actual_freq = controls.shape[1] / self.time_horizon
        
        freq_error = abs(actual_freq - target_freq) / target_freq * 100
        return freq_error
    
    def _check_collisions(self, trajectory: np.ndarray, obstacles: List[Dict]) -> bool:
        """Check if trajectory collides with obstacles"""
        for point in trajectory:
            for obstacle in obstacles:
                if np.linalg.norm(point[:2] - obstacle['position'][:2]) < obstacle['radius']:
                    return True
        return False
    
    def _check_safety_margins(self, trajectory: np.ndarray, other_vehicles: List[Dict]) -> int:
        """Count safety margin violations"""
        violations = 0
        
        for point in trajectory:
            for vehicle in other_vehicles:
                distance = np.linalg.norm(point[:2] - vehicle['position'][:2])
                if distance < self.safety_margin:
                    violations += 1
                    break
        
        return violations
    
    def _compute_comfort_score(self, trajectory: np.ndarray, controls: np.ndarray) -> float:
        """Compute passenger comfort score"""
        # Acceleration-based comfort
        if len(trajectory) > 2:
            velocities = np.diff(trajectory, axis=0) * 10  # Convert to m/s
            accelerations = np.diff(velocities, axis=0)
            
            # Compute lateral and longitudinal acceleration
            lat_accel = np.abs(accelerations[:, 1])
            lon_accel = np.abs(accelerations[:, 0])
            
            # Comfort score based on acceleration magnitude
            avg_accel = np.mean(np.sqrt(lat_accel**2 + lon_accel**2))
            comfort_score = max(0, 100 - avg_accel * 10)  # 10 points per m/s²
        else:
            comfort_score = 100.0
        
        # Penalize harsh braking
        harsh_brake_penalty = np.sum(np.maximum(controls[:, 2] - 0.7, 0)) * 10
        comfort_score -= harsh_brake_penalty
        
        return max(0, min(100, comfort_score))
    
    def _controls_to_waypoints(self, controls: torch.Tensor) -> torch.Tensor:
        """Convert control sequence to waypoints"""
        B, T, _ = controls.shape
        waypoints = torch.zeros(B, T, 3, device=controls.device)
        
        for b in range(B):
            for t in range(T):
                if t == 0:
                    waypoints[b, t, :2] = 0.0  # Start at origin
                    waypoints[b, t, 2] = 0.0   # z = 0
                else:
                    # Simple kinematic integration
                    dt = 0.1  # 100ms timestep
                    steering = controls[b, t-1, 0]
                    throttle = torch.clamp(controls[b, t-1, 1], 0, 1)
                    brake = torch.clamp(controls[b, t-1, 2], 0, 1)
                    
                    speed = throttle * 15.0 - brake * 5.0  # Simplified speed model
                    speed = torch.clamp(speed, 0, 30.0)
                    
                    # Update position (simplified)
                    dx = speed * dt * torch.cos(steering)
                    dy = speed * dt * torch.sin(steering)
                    
                    waypoints[b, t, 0] = waypoints[b, t-1, 0] + dx
                    waypoints[b, t, 1] = waypoints[b, t-1, 1] + dy
                    waypoints[b, t, 2] = 0.0  # z = 0
        
        return waypoints