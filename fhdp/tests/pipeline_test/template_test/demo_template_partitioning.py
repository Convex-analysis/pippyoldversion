#!/usr/bin/env python3
"""
Model Partitioning Demo Script

This script demonstrates how FHDP uses templates to partition models
across multiple vehicles in a pipeline.

Features:
1. Template-based model partitioning
2. Communication pattern visualization
3. Fragment validation
4. Performance benchmarking
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from fhdp.core.types import (
    PipelineTemplate, ResourceClass, TrainingConfig
)
from fhdp.edge_server.template_manager import TemplateManager


# ==================== Demo Model ====================

class DemoCNN(nn.Module):
    """
    CNN model designed for pipeline partitioning
    
    Architecture:
    - Stage 1: Conv1 -> Conv2 -> Pool (Feature extraction)
    - Stage 2: Conv3 -> Conv4 -> Pool (Deep features)
    - Stage 3: FC1 -> FC2 -> FC3 (Classification)
    """
    
    def __init__(self):
        super().__init__()
        
        # Stage 1: Early convolution
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Stage 2: Deep convolution
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Stage 3: Classification
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Stage 1
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool1(torch.relu(self.bn2(self.conv2(x))))
        
        # Stage 2
        x = self.pool2(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool2(torch.relu(self.bn4(self.conv4(x))))
        
        # Stage 3
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.bn5(self.fc1(x))))
        x = torch.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def get_stage_parameters(self, stage: int) -> Dict[str, nn.Parameter]:
        """Get parameters for a specific stage"""
        if stage == 0:
            # Stage 1: Early convolution layers
            return {
                'conv1.weight': self.conv1.weight,
                'conv1.bias': self.conv1.bias,
                'bn1.weight': self.bn1.weight,
                'bn1.bias': self.bn1.bias,
                'conv2.weight': self.conv2.weight,
                'conv2.bias': self.conv2.bias,
                'bn2.weight': self.bn2.weight,
                'bn2.bias': self.bn2.bias
            }
        elif stage == 1:
            # Stage 2: Deep convolution layers
            return {
                'conv3.weight': self.conv3.weight,
                'conv3.bias': self.conv3.bias,
                'bn3.weight': self.bn3.weight,
                'bn3.bias': self.bn3.bias,
                'conv4.weight': self.conv4.weight,
                'conv4.bias': self.conv4.bias,
                'bn4.weight': self.bn4.weight,
                'bn4.bias': self.bn4.bias
            }
        elif stage == 2:
            # Stage 3: Classification layers
            return {
                'fc1.weight': self.fc1.weight,
                'fc1.bias': self.fc1.bias,
                'bn5.weight': self.bn5.weight,
                'bn5.bias': self.bn5.bias,
                'fc2.weight': self.fc2.weight,
                'fc2.bias': self.fc2.bias,
                'bn6.weight': self.bn6.weight,
                'bn6.bias': self.bn6.bias,
                'fc3.weight': self.fc3.weight,
                'fc3.bias': self.fc3.bias
            }
        return {}


# ==================== Model Partitioner ====================

class TemplateBasedPartitioner:
    """Partition model based on pipeline template"""
    
    def __init__(self, template: PipelineTemplate):
        self.template = template
        self.fragment_size = template.model_fragment_size
        self.num_stages = len(template.resource_requirements)
        self.comm_pattern = template.communication_pattern
    
    def partition_model(self, model: nn.Module) -> List[Dict[str, nn.Parameter]]:
        """
        Partition model into fragments based on template
        
        Args:
            model: PyTorch model to partition
            
        Returns:
            List of parameter dictionaries, one per stage
        """
        fragments = []
        
        # Method 1: Use model's stage definitions if available
        if hasattr(model, 'get_stage_parameters'):
            for stage in range(self.num_stages):
                fragment = model.get_stage_parameters(stage)
                fragments.append(fragment)
        else:
            # Method 2: Generic partitioning
            fragments = self._generic_partition(model)
        
        return fragments
    
    def _generic_partition(self, model: nn.Module) -> List[Dict[str, nn.Parameter]]:
        """Generic partitioning by parameter count"""
        all_params = list(model.named_parameters())
        
        # Calculate parameters per fragment
        total_params = sum(p.numel() for _, p in all_params)
        params_per_fragment = total_params // self.num_stages
        
        fragments = []
        current_fragment = {}
        current_count = 0
        fragment_idx = 0
        
        for name, param in all_params:
            current_fragment[name] = param
            current_count += param.numel()
            
            # Check if fragment is full
            if current_count >= params_per_fragment and fragment_idx < self.num_stages - 1:
                fragments.append(current_fragment)
                current_fragment = {}
                current_count = 0
                fragment_idx += 1
        
        # Add remaining to last fragment
        if current_fragment:
            fragments.append(current_fragment)
        
        # Ensure correct number of fragments
        while len(fragments) < self.num_stages:
            fragments.append({})
        
        return fragments
    
    def get_fragment_info(self, fragments: List[Dict[str, nn.Parameter]]) -> List[Dict[str, any]]:
        """Get detailed information about each fragment"""
        fragment_info = []
        
        for i, fragment in enumerate(fragments):
            param_count = sum(p.numel() for p in fragment.values())
            param_size_mb = sum(p.numel() * p.element_size() for p in fragment.values()) / (1024 * 1024)
            layer_names = list(fragment.keys())
            
            fragment_info.append({
                'stage': i,
                'parameter_count': param_count,
                'size_mb': param_size_mb,
                'layers': layer_names,
                'layer_count': len(layer_names)
            })
        
        return fragment_info
    
    def visualize_communication_pattern(self) -> str:
        """Visualize communication pattern as text diagram"""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("Communication Pattern (from Template)")
        lines.append("=" * 60)
        
        # Create adjacency matrix
        n = self.num_stages
        matrix = np.zeros((n, n), dtype=int)
        
        for from_stage, to_stage in self.comm_pattern:
            matrix[from_stage][to_stage] = 1
            matrix[to_stage][from_stage] = 1  # Bidirectional
        
        # Print matrix
        lines.append("\nAdjacency Matrix:")
        lines.append("  " + "  ".join(f"S{i}" for i in range(n)))
        for i in range(n):
            line = f"S{i} "
            for j in range(n):
                line += " 1 " if matrix[i][j] else " 0 "
            lines.append(line)
        
        # Print connections
        lines.append("\nDirect Connections:")
        for from_stage, to_stage in sorted(self.comm_pattern):
            lines.append(f"  Stage {from_stage} <---> Stage {to_stage}")
        
        # Calculate connectivity
        connections_per_stage = [sum(matrix[i]) for i in range(n)]
        lines.append(f"\nConnectivity per Stage:")
        for i, count in enumerate(connections_per_stage):
            lines.append(f"  Stage {i}: {count} connections")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def validate_partitioning(self, fragments: List[Dict[str, nn.Parameter]]) -> Tuple[bool, List[str]]:
        """
        Validate model partitioning
        
        Returns:
            (is_valid, issues)
        """
        issues = []
        
        # Check fragment count
        if len(fragments) != self.num_stages:
            issues.append(f"Expected {self.num_stages} fragments, got {len(fragments)}")
        
        # Check for empty fragments
        for i, fragment in enumerate(fragments):
            if not fragment:
                issues.append(f"Fragment {i} is empty")
        
        # Check communication pattern
        for from_stage, to_stage in self.comm_pattern:
            if from_stage >= self.num_stages or to_stage >= self.num_stages:
                issues.append(f"Invalid communication pattern: ({from_stage}, {to_stage})")
        
        # Check parameter balance
        param_counts = [sum(p.numel() for p in f.values()) for f in fragments]
        if param_counts:
            avg_count = np.mean(param_counts)
            for i, count in enumerate(param_counts):
                if abs(count - avg_count) / avg_count > 0.7:  # 70% deviation
                    issues.append(f"Fragment {i} has {count} params, average is {avg_count:.0f}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


# ==================== Demo Functions ====================

def demo_2_stage_partition():
    """Demonstrate 2-stage pipeline partitioning"""
    print("\n" + "=" * 70)
    print("DEMO 1: 2-Stage Pipeline Partitioning")
    print("=" * 70)
    
    # Create template for 2-stage pipeline
    template = PipelineTemplate(
        template_id="demo_2_stage",
        resource_requirements=[ResourceClass.HIGH, ResourceClass.HIGH],
        expected_duration=10.0,
        communication_pattern=[(0, 1)],
        training_config=TrainingConfig(epochs=2, batch_size=32),
        model_fragment_size=50 * 1024 * 1024  # 50MB per fragment
    )
    
    # Create and partition model
    model = DemoCNN()
    partitioner = TemplateBasedPartitioner(template)
    fragments = partitioner.partition_model(model)
    
    # Get fragment info
    fragment_info = partitioner.get_fragment_info(fragments)
    
    print("\nFragment Information:")
    print("-" * 70)
    for info in fragment_info:
        print(f"\nStage {info['stage']}:")
        print(f"  Layers: {info['layer_count']}")
        print(f"  Parameters: {info['parameter_count']:,}")
        print(f"  Size: {info['size_mb']:.2f} MB")
        print(f"  Layer names: {', '.join(info['layers'][:3])}")
        if len(info['layers']) > 3:
            print(f"              ... and {len(info['layers']) - 3} more")
    
    # Validate
    is_valid, issues = partitioner.validate_partitioning(fragments)
    print(f"\nValidation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Visualize communication
    print(partitioner.visualize_communication_pattern())


def demo_3_stage_partition():
    """Demonstrate 3-stage pipeline partitioning"""
    print("\n" + "=" * 70)
    print("DEMO 2: 3-Stage Pipeline Partitioning")
    print("=" * 70)
    
    # Create template for 3-stage pipeline
    template = PipelineTemplate(
        template_id="demo_3_stage",
        resource_requirements=[ResourceClass.HIGH, ResourceClass.MEDIUM, ResourceClass.HIGH],
        expected_duration=15.0,
        communication_pattern=[(0, 1), (1, 2), (0, 2)],
        training_config=TrainingConfig(epochs=2, batch_size=32),
        model_fragment_size=33 * 1024 * 1024  # 33MB per fragment
    )
    
    # Create and partition model
    model = DemoCNN()
    partitioner = TemplateBasedPartitioner(template)
    fragments = partitioner.partition_model(model)
    
    # Get fragment info
    fragment_info = partitioner.get_fragment_info(fragments)
    
    print("\nFragment Information:")
    print("-" * 70)
    for info in fragment_info:
        print(f"\nStage {info['stage']} ({['AGX Orin (High)', 'Orin Nano (Medium)', 'AGX Orin (High)'][info['stage']]}):")
        print(f"  Layers: {info['layer_count']}")
        print(f"  Parameters: {info['parameter_count']:,}")
        print(f"  Size: {info['size_mb']:.2f} MB")
        print(f"  Layer names: {', '.join(info['layers'][:3])}")
        if len(info['layers']) > 3:
            print(f"              ... and {len(info['layers']) - 3} more")
    
    # Validate
    is_valid, issues = partitioner.validate_partitioning(fragments)
    print(f"\nValidation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Visualize communication
    print(partitioner.visualize_communication_pattern())


def demo_template_matching():
    """Demonstrate template matching from TemplateManager"""
    print("\n" + "=" * 70)
    print("DEMO 3: Template Matching Performance")
    print("=" * 70)
    
    # Initialize template manager
    manager = TemplateManager()
    
    # Create test vehicles
    from fhdp.core.types import VehicleInfo
    vehicles = [
        VehicleInfo(
            vehicle_id=f"vehicle_{i}",
            position=(i * 100.0, 0.0),
            velocity=10.0,
            direction=0.0,
            resources={
                'cpu': 0.8 if i < 3 else 0.6,
                'memory': 0.7 if i < 3 else 0.5,
                'battery': 0.8
            }
        )
        for i in range(5)
    ]
    
    print(f"\nCreated {len(vehicles)} test vehicles")
    for i, vehicle in enumerate(vehicles):
        print(f"  {vehicle.vehicle_id}: CPU={vehicle.resources['cpu']:.2f}, "
              f"Memory={vehicle.resources['memory']:.2f}")
    
    # Find template for different vehicle combinations
    print("\nFinding templates for vehicle combinations:")
    print("-" * 70)
    
    for num_vehicles in [2, 3, 4]:
        selected_vehicles = vehicles[:num_vehicles]
        vehicle_ids = [v.vehicle_id for v in selected_vehicles]
        
        start_time = time.time()
        template = manager.find_template_for_vehicles(selected_vehicles)
        lookup_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"\nVehicles: {', '.join(vehicle_ids)}")
        print(f"  Template found: {template.template_id if template else 'None'}")
        print(f"  Lookup time: {lookup_time:.2f} ms")
        
        if template:
            print(f"  Stages: {len(template.resource_requirements)}")
            print(f"  Expected duration: {template.expected_duration:.1f}s")
            print(f"  Resource requirements: {[rc.value for rc in template.resource_requirements]}")


def demo_performance_benchmark():
    """Benchmark partitioning performance"""
    print("\n" + "=" * 70)
    print("DEMO 4: Performance Benchmarking")
    print("=" * 70)
    
    # Create templates
    templates = [
        PipelineTemplate(
            template_id=f"bench_{i}",
            resource_requirements=[ResourceClass.HIGH, ResourceClass.MEDIUM, ResourceClass.HIGH],
            expected_duration=15.0,
            communication_pattern=[(0, 1), (1, 2), (0, 2)],
            training_config=TrainingConfig(epochs=2, batch_size=32),
            model_fragment_size=33 * 1024 * 1024
        )
        for i in range(100)
    ]
    
    # Model
    model = DemoCNN()
    
    # Benchmark partitioning
    print("\nBenchmarking model partitioning:")
    print("-" * 70)
    
    times = []
    for i, template in enumerate(templates):
        partitioner = TemplateBasedPartitioner(template)
        
        start_time = time.time()
        fragments = partitioner.partition_model(model)
        partition_time = time.time() - start_time
        
        times.append(partition_time)
        
        if i == 0 or i == 99:
            print(f"  Partition {i+1}: {partition_time*1000:.2f} ms")
    
    avg_time = np.mean(times) * 1000
    max_time = np.max(times) * 1000
    min_time = np.min(times) * 1000
    
    print(f"\nStatistics:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    
    # Benchmark template lookup
    manager = TemplateManager()
    from fhdp.core.types import VehicleInfo
    vehicles = [
        VehicleInfo(f"v{i}", (0, 0), 0, 0, {'cpu': 0.7, 'memory': 0.6, 'battery': 0.8})
        for i in range(5)
    ]
    
    print("\nBenchmarking template lookup:")
    print("-" * 70)
    
    lookup_times = []
    for _ in range(100):
        start_time = time.time()
        template = manager.find_template_for_vehicles(vehicles[:3])
        lookup_time = (time.time() - start_time) * 1000
        lookup_times.append(lookup_time)
    
    avg_lookup = np.mean(lookup_times)
    max_lookup = np.max(lookup_times)
    
    print(f"  Average: {avg_lookup:.2f} ms")
    print(f"  Max: {max_lookup:.2f} ms")
    print(f"  Target: <5.0 ms")
    print(f"  Compliant: {'✓ YES' if avg_lookup < 5.0 else '✗ NO'}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("FHDP Template-Based Model Partitioning Demo")
    print("=" * 70)
    print("\nThis demo shows how FHDP uses templates to partition")
    print("models across multiple vehicles in a pipeline.")
    
    # Run demos
    demo_2_stage_partition()
    demo_3_stage_partition()
    demo_template_matching()
    demo_performance_benchmark()
    
    # Summary
    print("\n" + "=" * 70)
    print("Demo Summary")
    print("=" * 70)
    print("\n✓ Demonstrated 2-stage pipeline partitioning")
    print("✓ Demonstrated 3-stage pipeline partitioning")
    print("✓ Showed template matching performance")
    print("✓ Benchmarking partitioning and lookup")
    print("\nKey Features:")
    print("  • Template-based model partitioning")
    print("  • Communication pattern visualization")
    print("  • Fragment validation")
    print("  • Performance <5ms for template lookup")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
