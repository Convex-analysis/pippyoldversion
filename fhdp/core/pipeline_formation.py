"""
Pipeline formation utilities for FHDP.

Includes pipeline formation logic and latency measurement for Level 2.
"""

from __future__ import annotations

import time
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import VehicleInfo, Pipeline, PipelineTemplate, ResourceClass
from .pipeline_model import get_pipeline_template

@dataclass
class PipelineFormationResult:
    """Result of pipeline formation"""
    pipeline: Optional[Pipeline] = None
    success: bool = False
    latency: float = 0.0
    error: Optional[str] = None

class PipelineFormationManager:
    """Manages pipeline formation process"""
    
    def __init__(self):
        self.latency_log_file = "level2_latency.csv"
        self._init_latency_log()
    
    def _init_latency_log(self) -> None:
        """Initialize latency log file"""
        try:
            with open(self.latency_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'latency_ms'])
        except Exception as e:
            print(f"Error initializing latency log: {e}")
    
    def _log_latency(self, latency: float) -> None:
        """Log latency to CSV file"""
        try:
            with open(self.latency_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), latency * 1000])  # Convert to milliseconds
        except Exception as e:
            print(f"Error logging latency: {e}")
    
    def form_pipeline(self, vehicles: List[VehicleInfo], template_id: Optional[str] = None) -> PipelineFormationResult:
        """Form a pipeline from available vehicles"""
        start_time = time.time()
        
        try:
            # Step 1: Validate input
            if not vehicles:
                return PipelineFormationResult(success=False, error="No vehicles provided")
            
            # Step 2: Get appropriate template
            default_template_id = "resnet18_2stage_high_high"  # Default template
            template = get_pipeline_template(template_id or default_template_id, default_template_id)
            
            # Step 3: Validate resource requirements
            resource_requirements = template.resource_requirements
            if len(vehicles) != len(resource_requirements):
                return PipelineFormationResult(
                    success=False, 
                    error=f"Mismatch in vehicle count and template requirements: {len(vehicles)} vs {len(resource_requirements)}"
                )
            
            # Step 4: Create pipeline
            pipeline = Pipeline(
                pipeline_id=f"pipeline_{int(time.time())}",
                vehicles=vehicles,
                stages=[f"stage{i}" for i in range(len(resource_requirements))],
                template_id=template.template_id,
                start_time=start_time,
                expected_completion=start_time + template.expected_duration,
                communication_pattern=template.communication_pattern,
                training_config=template.training_config
            )
            
            # Calculate latency
            latency = time.time() - start_time
            self._log_latency(latency)
            
            return PipelineFormationResult(
                pipeline=pipeline,
                success=True,
                latency=latency
            )
            
        except Exception as e:
            latency = time.time() - start_time
            self._log_latency(latency)
            return PipelineFormationResult(
                success=False,
                latency=latency,
                error=str(e)
            )
    
    def broadcast_pipeline_request(self, initiator: VehicleInfo, available_vehicles: List[VehicleInfo]) -> List[VehicleInfo]:
        """Broadcast pipeline formation request to available vehicles"""
        # In a real implementation, this would use network communication
        # For now, we'll simulate a simple selection process
        selected_vehicles = [initiator]
        
        # Select vehicles based on resource requirements
        for vehicle in available_vehicles:
            if vehicle.vehicle_id != initiator.vehicle_id:
                selected_vehicles.append(vehicle)
                if len(selected_vehicles) >= 3:  # Max 3 vehicles per pipeline
                    break
        
        return selected_vehicles
    
    def validate_pipeline(self, pipeline: Pipeline) -> bool:
        """Validate pipeline configuration"""
        # Check if all vehicles are available
        # Check if resource requirements are met
        # Check if communication pattern is valid
        return True
    
    def dissolve_pipeline(self, pipeline: Pipeline) -> bool:
        """Dissolve a pipeline"""
        # In a real implementation, this would notify all vehicles
        # For now, we'll just return success
        return True