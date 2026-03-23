#!/usr/bin/env python3
"""
Template Manager Test Suite for FHDP Pipeline Mode

This test suite validates the template management system's capabilities,
particularly focusing on model partitioning based on templates.

Tests cover:
1. Template generation from successful pipelines
2. Template matching and basket organization
3. Model fragmentation based on template
4. Communication pattern inference
5. Performance metrics (<5ms lookup, <1.5s composition)
"""

import sys
import os
import time
import unittest
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.types import (
    PipelineTemplate, Pipeline, VehicleInfo, ResourceClass,
    TrainingConfig, ModelUpdate, AggregationResult
)
from edge_server.template_manager import (
    TemplateManager, TemplateGenerator, TemplateMatcher,
    TemplateBasket
)


# ==================== Test Model ====================

class TestModel(nn.Module):
    """Simple CNN model for testing model partitioning"""
    
    def __init__(self):
        super().__init__()
        # Layer definitions with clear boundaries for partitioning
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Stage 1: Early feature extraction
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Stage 2: Deep feature extraction
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        
        # Stage 3: Classification
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_stage_params(self, stage: int) -> Dict[str, torch.Tensor]:
        """Get parameters for a specific stage"""
        if stage == 0:
            # Stage 1: Early convolution layers
            return {
                'conv1.weight': self.conv1.weight,
                'conv1.bias': self.conv1.bias,
                'conv2.weight': self.conv2.weight,
                'conv2.bias': self.conv2.bias
            }
        elif stage == 1:
            # Stage 2: Deep convolution layers
            return {
                'conv3.weight': self.conv3.weight,
                'conv3.bias': self.conv3.bias,
                'conv4.weight': self.conv4.weight,
                'conv4.bias': self.conv4.bias
            }
        elif stage == 2:
            # Stage 3: Fully connected layers
            return {
                'fc1.weight': self.fc1.weight,
                'fc1.bias': self.fc1.bias,
                'fc2.weight': self.fc2.weight,
                'fc2.bias': self.fc2.bias,
                'fc3.weight': self.fc3.weight,
                'fc3.bias': self.fc3.bias
            }
        return {}


# ==================== Model Partitioner ====================

class ModelPartitioner:
    """Model partitioning based on pipeline template"""
    
    def __init__(self, template: PipelineTemplate):
        self.template = template
        self.fragment_size = template.model_fragment_size
        self.comm_pattern = template.communication_pattern
        self.stages_count = len(template.resource_requirements)
    
    def partition_model(self, model: nn.Module) -> List[Dict[str, torch.Tensor]]:
        """
        Partition model into fragments based on template
        
        Args:
            model: PyTorch model to partition
            
        Returns:
            List of parameter dictionaries, one per stage
        """
        fragments = []
        
        # Strategy 1: Layer-based partitioning
        if hasattr(model, 'get_stage_params'):
            for stage in range(self.stages_count):
                fragment = model.get_stage_params(stage)
                fragments.append(fragment)
        else:
            # Strategy 2: Generic partitioning based on parameter count
            fragments = self._partition_by_parameters(model)
        
        return fragments
    
    def _partition_by_parameters(self, model: nn.Module) -> List[Dict[str, torch.Tensor]]:
        """Generic partitioning by parameter count"""
        all_params = list(model.named_parameters())
        
        # Calculate target size per fragment
        total_params = sum(p.numel() for _, p in all_params)
        params_per_fragment = total_params // self.stages_count
        
        fragments = []
        current_fragment = {}
        current_count = 0
        fragment_idx = 0
        
        for name, param in all_params:
            current_fragment[name] = param
            current_count += param.numel()
            
            # Check if fragment is full
            if current_count >= params_per_fragment and fragment_idx < self.stages_count - 1:
                fragments.append(current_fragment)
                current_fragment = {}
                current_count = 0
                fragment_idx += 1
        
        # Add remaining parameters to last fragment
        if current_fragment:
            fragments.append(current_fragment)
        
        # Ensure we have the right number of fragments
        while len(fragments) < self.stages_count:
            fragments.append({})
        
        return fragments
    
    def get_communication_graph(self) -> Dict[int, List[int]]:
        """
        Get communication graph from template
        
        Returns:
            Dictionary mapping stage -> list of connected stages
        """
        graph = defaultdict(list)
        
        for from_stage, to_stage in self.comm_pattern:
            graph[from_stage].append(to_stage)
            graph[to_stage].append(from_stage)  # Bidirectional
        
        return dict(graph)
    
    def validate_partitioning(self, fragments: List[Dict[str, torch.Tensor]]) -> Tuple[bool, str]:
        """
        Validate model partitioning
        
        Returns:
            (is_valid, message)
        """
        # Check fragment count
        if len(fragments) != self.stages_count:
            return False, f"Expected {self.stages_count} fragments, got {len(fragments)}"
        
        # Check for empty fragments
        for i, fragment in enumerate(fragments):
            if not fragment:
                return False, f"Fragment {i} is empty"
        
        # Check communication pattern consistency
        for from_stage, to_stage in self.comm_pattern:
            if from_stage >= self.stages_count or to_stage >= self.stages_count:
                return False, f"Invalid communication pattern: ({from_stage}, {to_stage})"
        
        return True, "Partitioning is valid"


# ==================== Test Cases ====================

class TestTemplateGeneration(unittest.TestCase):
    """Test template generation from pipelines"""
    
    def setUp(self):
        self.generator = TemplateGenerator()
    
    def test_template_from_pipeline(self):
        """Test generating template from a successful pipeline"""
        # Create test pipeline
        pipeline = Pipeline(
            pipeline_id="test_pipeline_001",
            template_id="temp_001",
            vehicles=["vehicle_1", "vehicle_2", "vehicle_3"],
            stages=["stage_0", "stage_1", "stage_2"],
            start_time=time.time() - 30.0,
            expected_completion=time.time()
        )
        
        # Generate template
        template = self.generator.generate_template_from_pipeline(pipeline, success_rate=0.95)
        
        # Validate template structure
        self.assertIsNotNone(template)
        self.assertEqual(len(template.resource_requirements), 3)
        self.assertIsInstance(template.resource_requirements[0], ResourceClass)
        self.assertGreater(template.expected_duration, 0)
        self.assertIsInstance(template.communication_pattern, list)
        self.assertIsInstance(template.training_config, TrainingConfig)
    
    def test_resource_pattern_extraction(self):
        """Test resource pattern extraction"""
        pipeline = Pipeline(
            pipeline_id="test_002",
            template_id="temp_002",
            vehicles=["v1", "v2", "v3", "v4"],
            stages=["s0", "s1", "s2", "s3"],
            start_time=time.time(),
            expected_completion=time.time() + 20.0
        )
        
        template = self.generator.generate_template_from_pipeline(pipeline, 1.0)
        resource_pattern = template.resource_requirements
        
        # Check that first and last are HIGH
        self.assertEqual(resource_pattern[0], ResourceClass.HIGH)
        self.assertEqual(resource_pattern[-1], ResourceClass.HIGH)
    
    def test_communication_pattern_inference(self):
        """Test communication pattern inference"""
        pipeline = Pipeline(
            pipeline_id="test_003",
            template_id="temp_003",
            vehicles=["v1", "v2", "v3"],
            stages=["s0", "s1", "s2"],
            start_time=time.time(),
            expected_completion=time.time() + 15.0
        )
        
        template = self.generator.generate_template_from_pipeline(pipeline, 1.0)
        comm_pattern = template.communication_pattern
        
        # Check for linear connections
        self.assertIn((0, 1), comm_pattern)
        self.assertIn((1, 2), comm_pattern)
    
    def test_synthetic_template_generation(self):
        """Test generation of synthetic templates"""
        templates = self.generator.generate_synthetic_templates(num_templates=20)
        
        # Check template count
        self.assertEqual(len(templates), 20)
        
        # Check template diversity
        lengths = set(len(t.resource_requirements) for t in templates)
        self.assertGreater(len(lengths), 1)  # Should have different lengths
        
        # Check resource variety
        resource_classes = set()
        for template in templates:
            for rc in template.resource_requirements:
                resource_classes.add(rc)
        self.assertEqual(resource_classes, {ResourceClass.HIGH, ResourceClass.MEDIUM, ResourceClass.LOW})


class TestTemplateMatching(unittest.TestCase):
    """Test template matching performance and accuracy"""
    
    def setUp(self):
        self.matcher = TemplateMatcher()
        self.test_vehicles = self._create_test_vehicles()
    
    def _create_test_vehicles(self) -> List[VehicleInfo]:
        """Create test vehicles with different resources"""
        return [
            VehicleInfo(
                vehicle_id=f"vehicle_{i}",
                position=(0.0, 0.0),
                velocity=0.0,
                direction=0.0,
                resources={
                    'cpu': 0.7 + (i % 3) * 0.1,
                    'memory': 0.6 + (i % 3) * 0.15,
                    'battery': 0.8 - (i % 3) * 0.1
                }
            )
            for i in range(10)
        ]
    
    def test_template_addition(self):
        """Test adding templates to matcher"""
        templates = TemplateGenerator().generate_synthetic_templates(10)
        
        for template in templates:
            self.matcher.add_template(template)
        
        # Check that templates were added
        self.assertGreater(len(self.matcher.baskets), 0)
    
    def test_template_lookup_performance(self):
        """Test that template lookup is <5ms"""
        # Add some templates
        generator = TemplateGenerator()
        templates = generator.generate_synthetic_templates(100)
        
        for template in templates:
            self.matcher.add_template(template)
        
        # Measure lookup time
        start_time = time.time()
        candidates = self.matcher.find_best_template(self.test_vehicles[:3])
        lookup_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should be <5ms
        self.assertLess(lookup_time, 5.0, f"Lookup took {lookup_time:.2f}ms")
    
    def test_template_caching(self):
        """Test that template caching improves performance"""
        # Add templates
        generator = TemplateGenerator()
        templates = generator.generate_synthetic_templates(50)
        
        for template in templates:
            self.matcher.add_template(template)
        
        # First lookup (cache miss)
        start_time = time.time()
        self.matcher.find_best_template(self.test_vehicles[:3])
        first_time = time.time() - start_time
        
        # Second lookup (cache hit)
        start_time = time.time()
        self.matcher.find_best_template(self.test_vehicles[:3])
        second_time = time.time() - start_time
        
        # Cached lookup should be faster
        self.assertLess(second_time, first_time)
    
    def test_basket_organization(self):
        """Test basket-based organization"""
        generator = TemplateGenerator()
        templates = generator.generate_synthetic_templates(50)
        
        for template in templates:
            self.matcher.add_template(template)
        
        # Check that templates are organized in baskets
        total_templates = sum(len(basket.templates) for basket in self.matcher.baskets.values())
        self.assertEqual(total_templates, 50)
        
        # Check that same-signature templates are in same basket
        for basket in self.matcher.baskets.values():
            if len(basket.templates) > 1:
                # All templates in basket should have same signature
                first_sig = basket.resource_signature
                for template in basket.templates:
                    sig = self.matcher._create_resource_signature(template.resource_requirements)
                    self.assertEqual(sig, first_sig)


class TestModelPartitioning(unittest.TestCase):
    """Test model partitioning based on templates"""
    
    def setUp(self):
        self.model = TestModel()
        self.test_templates = self._create_test_templates()
    
    def _create_test_templates(self) -> List[PipelineTemplate]:
        """Create test templates for partitioning"""
        return [
            PipelineTemplate(
                template_id="template_2_stage",
                resource_requirements=[ResourceClass.HIGH, ResourceClass.HIGH],
                expected_duration=10.0,
                communication_pattern=[(0, 1)],
                training_config=TrainingConfig(epochs=2, batch_size=32),
                model_fragment_size=50 * 1024 * 1024  # 50MB
            ),
            PipelineTemplate(
                template_id="template_3_stage",
                resource_requirements=[ResourceClass.HIGH, ResourceClass.MEDIUM, ResourceClass.HIGH],
                expected_duration=15.0,
                communication_pattern=[(0, 1), (1, 2), (0, 2)],
                training_config=TrainingConfig(epochs=2, batch_size=32),
                model_fragment_size=33 * 1024 * 1024  # 33MB
            )
        ]
    
    def test_model_partition_2_stage(self):
        """Test model partitioning for 2-stage pipeline"""
        template = self.test_templates[0]
        partitioner = ModelPartitioner(template)
        
        fragments = partitioner.partition_model(self.model)
        
        # Check fragment count
        self.assertEqual(len(fragments), 2)
        
        # Validate partitioning
        is_valid, message = partitioner.validate_partitioning(fragments)
        self.assertTrue(is_valid, message)
    
    def test_model_partition_3_stage(self):
        """Test model partitioning for 3-stage pipeline"""
        template = self.test_templates[1]
        partitioner = ModelPartitioner(template)
        
        fragments = partitioner.partition_model(self.model)
        
        # Check fragment count
        self.assertEqual(len(fragments), 3)
        
        # Validate partitioning
        is_valid, message = partitioner.validate_partitioning(fragments)
        self.assertTrue(is_valid, message)
    
    def test_communication_graph_extraction(self):
        """Test communication graph extraction from template"""
        template = self.test_templates[1]  # 3-stage template
        partitioner = ModelPartitioner(template)
        
        graph = partitioner.get_communication_graph()
        
        # Check graph structure
        self.assertIn(0, graph)
        self.assertIn(1, graph)
        self.assertIn(2, graph)
        
        # Check connections
        self.assertIn(1, graph[0])  # 0 -> 1
        self.assertIn(2, graph[1])  # 1 -> 2
        self.assertIn(0, graph[2])  # 0 -> 2 (skip connection)
        self.assertIn(2, graph[0])  # 2 -> 0 (bidirectional)
    
    def test_fragment_size_adherence(self):
        """Test that fragments respect template fragment size"""
        template = self.test_templates[0]
        partitioner = ModelPartitioner(template)
        
        fragments = partitioner.partition_model(self.model)
        
        # Calculate actual fragment sizes
        fragment_sizes = []
        for fragment in fragments:
            size = sum(p.numel() for p in fragment.values())
            fragment_sizes.append(size)
        
        # Check that fragments are roughly balanced
        avg_size = np.mean(fragment_sizes)
        for size in fragment_sizes:
            # Allow 50% deviation from average
            self.assertLess(abs(size - avg_size) / avg_size, 0.5)


class TestTemplateManagerIntegration(unittest.TestCase):
    """Test template manager integration with FHDP system"""
    
    def setUp(self):
        self.manager = TemplateManager()
        self.test_vehicles = self._create_test_vehicles()
    
    def _create_test_vehicles(self) -> List[VehicleInfo]:
        """Create test vehicles"""
        return [
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
    
    def test_template_discovery(self):
        """Test template discovery for vehicles"""
        # Find template for vehicles
        template = self.manager.find_template_for_vehicles(self.test_vehicles[:3])
        
        # Should find a template
        self.assertIsNotNone(template)
        self.assertIsInstance(template, PipelineTemplate)
    
    def test_pipeline_registration(self):
        """Test registering successful pipeline for learning"""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            template_id="temp_001",
            vehicles=["v1", "v2", "v3"],
            stages=["s0", "s1", "s2"],
            start_time=time.time() - 20.0,
            expected_completion=time.time()
        )
        
        # Register successful pipeline
        self.manager.register_successful_pipeline(pipeline, success=True, duration=20.0)
        
        # Check statistics
        stats = self.manager.get_template_statistics()
        self.assertGreater(stats['total_templates'], 100)  # Initial + new
    
    def test_template_learning(self):
        """Test that system learns from successful pipelines"""
        # Register several successful pipelines
        for i in range(5):
            pipeline = Pipeline(
                pipeline_id=f"pipeline_{i}",
                template_id=f"temp_{i}",
                vehicles=["v1", "v2"],
                stages=["s0", "s1"],
                start_time=time.time() - 15.0,
                expected_completion=time.time()
            )
            self.manager.register_successful_pipeline(pipeline, success=True, duration=15.0)
        
        # Check that new templates were generated
        stats = self.manager.get_template_statistics()
        self.assertGreaterEqual(stats['total_templates'], 105)  # Initial + new
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        stats = self.manager.get_template_statistics()
        
        # Check required statistics
        self.assertIn('total_baskets', stats)
        self.assertIn('total_templates', stats)
        self.assertIn('avg_success_rate', stats)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('memory_usage', stats)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics compliance"""
    
    def test_template_lookup_latency(self):
        """Test that template lookup latency is <5ms"""
        manager = TemplateManager()
        vehicles = [
            VehicleInfo(f"v{i}", (0, 0), 0, 0, 
                      {'cpu': 0.7, 'memory': 0.6, 'battery': 0.8})
            for i in range(5)
        ]
        
        # Measure multiple lookups
        times = []
        for _ in range(100):
            start_time = time.time()
            manager.find_template_for_vehicles(vehicles[:3])
            times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        # All lookups should be <5ms
        self.assertLess(avg_time, 5.0, f"Average lookup time: {avg_time:.2f}ms")
        self.assertLess(max_time, 5.0, f"Max lookup time: {max_time:.2f}ms")
    
    def test_pipeline_composition_time(self):
        """Test that pipeline composition is <1.5s"""
        manager = TemplateManager()
        vehicles = [
            VehicleInfo(f"v{i}", (i*100, 0), 10.0, 0.0,
                      {'cpu': 0.8, 'memory': 0.7, 'battery': 0.9})
            for i in range(3)
        ]
        
        # Measure composition time
        start_time = time.time()
        
        # Step 1: Find template
        template = manager.find_template_for_vehicles(vehicles)
        
        # Step 2: Partition model
        if template:
            partitioner = ModelPartitioner(template)
            model = TestModel()
            fragments = partitioner.partition_model(model)
            partitioner.validate_partitioning(fragments)
        
        composition_time = time.time() - start_time
        
        # Should be <1.5s
        self.assertLess(composition_time, 1.5, f"Composition took {composition_time:.2f}s")
    
    def test_memory_efficiency(self):
        """Test that template system uses <50MB memory"""
        manager = TemplateManager()
        
        # Get memory usage
        stats = manager.get_template_statistics()
        memory_bytes = stats['memory_usage']
        memory_mb = memory_bytes / (1024 * 1024)
        
        # Should be <50MB
        self.assertLess(memory_mb, 50.0, f"Memory usage: {memory_mb:.2f}MB")


# ==================== Test Runner ====================

def run_tests():
    """Run all template manager tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTemplateGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestTemplateMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPartitioning))
    suite.addTests(loader.loadTestsFromTestCase(TestTemplateManagerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
