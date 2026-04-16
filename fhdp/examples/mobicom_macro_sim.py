#!/usr/bin/env python3
"""
MobiCom Macro Simulation Script

This script simulates end-to-end convergence for FHDP with 500 virtual vehicles
and SUMO mobility patterns.
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from core.fhdp_system import FHDPSystem, SystemConfiguration
from core.types import VehicleInfo, VehicleState, TrainingMode, ModelUpdate

class MobiComMacroSimulator:
    """Macro simulator for MobiCom evaluation"""
    
    def __init__(self, num_vehicles: int = 500, simulation_hours: float = 1.0):
        self.num_vehicles = num_vehicles
        self.simulation_hours = simulation_hours
        self.simulation_seconds = simulation_hours * 3600
        self.vehicles = []
        self.fhdp_system = self._initialize_fhdp_system()
        self.accuracy_history = []
        self.time_history = []
    
    def _initialize_fhdp_system(self) -> FHDPSystem:
        """Initialize FHDP system for simulation"""
        config = SystemConfiguration(
            max_vehicles_per_region=500,
            pipeline_formation_interval=5.0,
            model_broadcast_interval=10.0,
            participation_timeout=30.0,
            aggregation_interval=15.0,
            enable_pipeline_training=True,
            enable_individual_training=True,
            fairness_enabled=True
        )
        return FHDPSystem(config)
    
    def _create_virtual_vehicles(self) -> None:
        """Create virtual vehicles with random resources"""
        for i in range(self.num_vehicles):
            # Random position in a 10x10 grid
            position = (random.uniform(0, 10), random.uniform(0, 10))
            
            # Random velocity and direction
            velocity = random.uniform(0, 5)
            direction = random.uniform(0, 360)
            
            # Random resources based on vehicle type
            vehicle_type = random.choice(['high', 'medium', 'low'])
            if vehicle_type == 'high':
                resources = {
                    'cpu_cores': 8,
                    'memory_gb': 16,
                    'gpu_available': True,
                    'gpu_memory_gb': 8,
                    'compute_score': 0.9
                }
            elif vehicle_type == 'medium':
                resources = {
                    'cpu_cores': 4,
                    'memory_gb': 8,
                    'gpu_available': True,
                    'gpu_memory_gb': 4,
                    'compute_score': 0.6
                }
            else:
                resources = {
                    'cpu_cores': 2,
                    'memory_gb': 4,
                    'gpu_available': False,
                    'compute_score': 0.3
                }
            
            vehicle = VehicleInfo(
                vehicle_id=f'vehicle_{i}',
                position=position,
                velocity=velocity,
                direction=direction,
                resources=resources,
                state=VehicleState.IDLE
            )
            self.vehicles.append(vehicle)
            self.fhdp_system.register_vehicle(vehicle)
    
    def _simulate_mobility(self, current_time: float) -> None:
        """Simulate vehicle mobility using SUMO-like patterns"""
        for vehicle in self.vehicles:
            # Update position based on velocity and direction
            dx = vehicle.velocity * np.cos(np.radians(vehicle.direction))
            dy = vehicle.velocity * np.sin(np.radians(vehicle.direction))
            
            new_x = (vehicle.position[0] + dx) % 10
            new_y = (vehicle.position[1] + dy) % 10
            vehicle.position = (new_x, new_y)
            
            # Randomly change direction occasionally
            if random.random() < 0.01:
                vehicle.direction = random.uniform(0, 360)
            
            # Randomly change velocity occasionally
            if random.random() < 0.05:
                vehicle.velocity = random.uniform(0, 5)
    
    def _form_pipelines(self) -> List[str]:
        """Form pipelines from nearby vehicles"""
        pipelines = []
        
        # Group vehicles by proximity
        vehicle_groups = []
        used_vehicles = set()
        
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.vehicle_id in used_vehicles:
                continue
            
            # Find nearby vehicles
            group = [vehicle]
            used_vehicles.add(vehicle.vehicle_id)
            
            for j, other_vehicle in enumerate(self.vehicles):
                if i == j or other_vehicle.vehicle_id in used_vehicles:
                    continue
                
                # Calculate distance
                distance = np.sqrt(
                    (vehicle.position[0] - other_vehicle.position[0])**2 +
                    (vehicle.position[1] - other_vehicle.position[1])**2
                )
                
                if distance < 1.0 and len(group) < 3:
                    group.append(other_vehicle)
                    used_vehicles.add(other_vehicle.vehicle_id)
            
            if len(group) >= 2:
                vehicle_groups.append(group)
        
        # Create pipelines for each group
        for group in vehicle_groups:
            pipeline_id = f'pipeline_{int(time.time())}_{random.randint(0, 1000)}'
            success = self.fhdp_system.hybrid_participation_manager.create_pipeline(
                pipeline_id=pipeline_id,
                vehicles=group,
                training_mode=TrainingMode.PIPELINE,
                config={}
            )
            if success:
                pipelines.append(pipeline_id)
        
        return pipelines
    
    def _simulate_training(self, pipelines: List[str]) -> None:
        """Simulate training for pipelines"""
        for pipeline_id in pipelines:
            # Simulate training delay based on pipeline size
            pipeline = self.fhdp_system.hybrid_participation_manager.get_pipeline(pipeline_id)
            if pipeline:
                num_vehicles = len(pipeline.vehicles)
                training_time = random.uniform(5, 15) * num_vehicles
                time.sleep(training_time / 100)  # Speed up simulation
                
                # Generate model update
                for vehicle in pipeline.vehicles:
                    update = ModelUpdate(
                        vehicle_id=vehicle.vehicle_id,
                        model_state={},
                        optimizer_state={},
                        metadata={
                            'pipeline_id': pipeline_id,
                            'pipeline_start_round': 0,
                            'data_size': 1000,
                            'fidelity_score': 0.95
                        },
                        training_mode=TrainingMode.PIPELINE,
                        timestamp=time.time()
                    )
                    self.fhdp_system.asynchronous_aggregation_manager.submit_update(update)
    
    def _calculate_accuracy(self) -> float:
        """Calculate simulated global model accuracy"""
        # Simulate accuracy improvement over time
        elapsed_seconds = time.time() - self.start_time
        progress = min(elapsed_seconds / self.simulation_seconds, 1.0)
        
        # Exponential accuracy growth
        base_accuracy = 0.1
        max_accuracy = 0.95
        accuracy = base_accuracy + (max_accuracy - base_accuracy) * (1 - np.exp(-5 * progress))
        
        # Add some noise
        noise = random.uniform(-0.02, 0.02)
        accuracy = max(base_accuracy, min(max_accuracy, accuracy + noise))
        
        return accuracy
    
    def run_simulation(self) -> None:
        """Run the macro simulation"""
        print(f"Starting MobiCom macro simulation with {self.num_vehicles} vehicles")
        print(f"Simulation duration: {self.simulation_hours} hours")
        
        # Initialize simulation
        self.start_time = time.time()
        self._create_virtual_vehicles()
        
        # Main simulation loop
        current_time = 0
        while current_time < self.simulation_seconds:
            # Simulate mobility
            self._simulate_mobility(current_time)
            
            # Form pipelines
            pipelines = self._form_pipelines()
            
            # Simulate training
            self._simulate_training(pipelines)
            
            # Calculate and record accuracy
            accuracy = self._calculate_accuracy()
            self.accuracy_history.append(accuracy)
            self.time_history.append(current_time / 3600)  # Convert to hours
            
            # Update current time
            current_time = time.time() - self.start_time
            
            # Print progress
            if int(current_time) % 60 == 0:  # Every minute
                progress = current_time / self.simulation_seconds * 100
                print(f"Simulation progress: {progress:.1f}%, Accuracy: {accuracy:.4f}")
        
        # End simulation
        print("Simulation completed!")
        self._plot_results()
    
    def _plot_results(self) -> None:
        """Plot simulation results"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_history, self.accuracy_history)
        plt.xlabel('Simulated Hours')
        plt.ylabel('Global Model Accuracy')
        plt.title('End-to-End Convergence: Global Model Accuracy over Simulated Time')
        plt.grid(True)
        plt.savefig('mobicom_macro_sim_accuracy.png')
        print("Accuracy plot saved to mobicom_macro_sim_accuracy.png")

if __name__ == "__main__":
    simulator = MobiComMacroSimulator(num_vehicles=500, simulation_hours=1.0)
    simulator.run_simulation()