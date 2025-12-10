#!/usr/bin/env python3
"""
FHDP System Test Suite

Comprehensive test suite for FHDP system components and integration.
"""
import unittest
import time
import torch
import numpy as np
from unittest.mock import Mock, patch

import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import *
from core.constants import *
from edge_server import *
from vehicle_layer import *
from core import FHDPSystem, SystemConfiguration

class TestMobilityPredictor(unittest.TestCase):
    """Test mobility prediction functionality"""
    
    def setUp(self):
        self.predictor = MobilityPredictor()
        self.vehicle_info = VehicleInfo(
            vehicle_id="test_vehicle",
            position=(100, 200),
            velocity=20.0,
            direction=0.5,
            resources={'cpu': 0.7, 'memory': 0.6}
        )
    
    def test_mobility_update(self):
        """Test mobility data update"""
        initial_predictions = len(self.predictor.predict_mobility("test_vehicle"))
        
        # Update vehicle mobility
        self.predictor.update_vehicle_mobility(self.vehicle_info)
        
        # Should be able to make predictions after update
        predictions = self.predictor.predict_mobility("test_vehicle")
        self.assertGreater(len(predictions), initial_predictions)
    
    def test_pipeline_stability_prediction(self):
        """Test pipeline stability prediction"""
        vehicle_ids = ["v1", "v2", "v3"]
        
        # Update mobility for all vehicles
        for vid in vehicle_ids:
            v_info = VehicleInfo(vid, (0, 0), 20, 0, {})
            self.predictor.update_vehicle_mobility(v_info)
        
        # Test stability prediction
        stability = self.predictor.predict_pipeline_stability(vehicle_ids, 10.0)
        self.assertIsInstance(stability, float)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)

class TestTemplateManager(unittest.TestCase):
    """Test template management functionality"""
    
    def setUp(self):
        self.manager = TemplateManager()
        
    def test_template_lookup_performance(self):
        """Test template lookup meets latency requirements"""
        # Create test vehicles
        vehicles = []
        for i in range(10):
            v_info = VehicleInfo(
                vehicle_id=f"test_{i}",
                position=(i*10, 0),
                velocity=20.0,
                direction=0.0,
                resources={'cpu': 0.7, 'memory': 0.6, 'battery': 0.8}
            )
            vehicles.append(v_info)
        
        # Test lookup performance
        start_time = time.time()
        template = self.manager.find_template_for_vehicles(vehicles)
        end_time = time.time()
        
        lookup_time = end_time - start_time
        self.assertLess(lookup_time, TEMPLATE_LOOKUP_LATENCY_THRESHOLD)
    
    def test_template_basket_organization(self):
        """Test template basket-based organization"""
        stats = self.manager.get_template_statistics()
        
        # Should have initial templates
        self.assertGreater(stats['total_templates'], 0)
        self.assertGreater(stats['total_baskets'], 0)

class TestResourceClassifier(unittest.TestCase):
    """Test resource classification functionality"""
    
    def setUp(self):
        self.classifier = ResourceClassifier()
        self.vehicle_info = VehicleInfo(
            vehicle_id="test_vehicle",
            position=(0, 0),
            velocity=0,
            direction=0,
            resources={'cpu': 0.9, 'memory': 0.8, 'battery': 0.9}
        )
    
    def test_high_resource_classification(self):
        """Test high resource vehicle classification"""
        resource_class = self.classifier.classify_vehicle(self.vehicle_info)
        self.assertEqual(resource_class, ResourceClass.HIGH)
    
    def test_fairness_metrics(self):
        """Test fairness metrics calculation"""
        # Record participation
        self.classifier.record_training_participation(
            "test_vehicle", TrainingMode.INDIVIDUAL, True, 1.0
        )
        
        # Get fairness metrics
        metrics = self.classifier.get_fairness_metrics("test_vehicle")
        self.assertEqual(metrics.vehicle_id, "test_vehicle")
        self.assertEqual(metrics.participation_count, 1)
        self.assertGreater(metrics.contribution_score, 0.0)

class TestPipelineFormation(unittest.TestCase):
    """Test pipeline formation functionality"""
    
    def setUp(self):
        self.vehicle_info = VehicleInfo("test_vehicle", (0, 0), 20, 0, {})
        self.formation = PipelineFormation(self.vehicle_info)
        
    def test_greedy_selection(self):
        """Test greedy selection algorithm"""
        from core.types import PipelineTemplate, TrainingConfig
        
        # Create test template
        template = PipelineTemplate(
            template_id="test_template",
            resource_requirements=[ResourceClass.MEDIUM] * 3,
            expected_duration=15.0,
            communication_pattern=[(0, 1), (1, 2)],
            training_config=TrainingConfig()
        )
        
        # Create test vehicles
        vehicles = []
        for i in range(5):
            v_info = VehicleInfo(
                vehicle_id=f"test_{i}",
                position=(i*10, 0),
                velocity=20.0,
                direction=0.0,
                resources={'cpu': 0.7, 'memory': 0.6, 'battery': 0.8}
            )
            vehicles.append(v_info)
        
        # Test pipeline formation
        pipeline_id = self.formation.initiate_pipeline_formation(template, vehicles)
        
        if pipeline_id:
            self.assertIsInstance(pipeline_id, str)
            pipelines = self.formation.get_active_pipelines()
            self.assertIn(pipeline_id, pipelines)

class TestTrainingExecutor(unittest.TestCase):
    """Test training execution functionality"""
    
    def setUp(self):
        self.vehicle_info = VehicleInfo("test_vehicle", (0, 0), 0, 0, {})
        self.executor = TrainingExecutor(self.vehicle_info)
    
    def test_communication_optimization(self):
        """Test communication optimization"""
        from core.types import ModelUpdate
        
        # Create test model update
        model_update = ModelUpdate(
            source_id="test_vehicle",
            update_data=torch.randn(1000),
            metadata={'data_size': 1000},
            timestamp=time.time()
        )
        
        # Test communication preparation
        communication_data = self.executor.prepare_communication(model_update)
        self.assertIsNotNone(communication_data)
    
    def test_lazy_error_propagation(self):
        """Test lazy error propagation"""
        error_signal = torch.randn(100)
        
        # Accumulate error
        self.executor.error_propagation.accumulate_error("test", error_signal, ["target1"])
        
        # Check if propagation should occur
        should_propagate = self.executor.error_propagation.should_propagate_errors()
        self.assertIsInstance(should_propagate, bool)

class TestFHDPSystem(unittest.TestCase):
    """Test FHDP system integration"""
    
    def setUp(self):
        self.config = SystemConfiguration(
            max_vehicles_per_region=10,
            pipeline_formation_interval=5.0,
            enable_pipeline_training=True,
            enable_individual_training=True
        )
        self.system = FHDPSystem(self.config)
    
    def test_system_lifecycle(self):
        """Test system start/stop lifecycle"""
        # Start system
        self.system.start_system()
        self.assertTrue(self.system.system_active)
        
        # Stop system
        self.system.stop_system()
        self.assertFalse(self.system.system_active)
    
    def test_vehicle_registration(self):
        """Test vehicle registration/unregistration"""
        vehicle_info = VehicleInfo(
            vehicle_id="test_vehicle",
            position=(100, 100),
            velocity=20.0,
            direction=0.0,
            resources={'cpu': 0.7, 'memory': 0.6, 'battery': 0.8}
        )
        
        # Register vehicle
        success = self.system.register_vehicle(vehicle_info)
        self.assertTrue(success)
        self.assertIn("test_vehicle", self.system.registered_vehicles)
        
        # Unregister vehicle
        self.system.unregister_vehicle("test_vehicle")
        self.assertNotIn("test_vehicle", self.system.registered_vehicles)
    
    def test_hybrid_participation(self):
        """Test hybrid participation model"""
        vehicle_info = VehicleInfo(
            vehicle_id="test_vehicle",
            position=(0, 0),
            velocity=20.0,
            direction=0.0,
            resources={'cpu': 0.9, 'memory': 0.8, 'battery': 0.9}
        )
        
        self.system.register_vehicle(vehicle_info)
        
        # Test participation decision
        decision = self.system.participation_manager.make_participation_decision(
            "test_vehicle",
            self.system.resource_classifier,
            self.system.template_manager
        )
        
        self.assertIn(decision, [TrainingMode.INDIVIDUAL, TrainingMode.PIPELINE])

class TestPerformanceRequirements(unittest.TestCase):
    """Test FHDP system performance requirements"""
    
    def setUp(self):
        self.manager = TemplateManager()
        self.formation = PipelineFormation(VehicleInfo("test", (0, 0), 0, 0, {}))
    
    def test_template_lookup_latency(self):
        """Test template lookup <5ms latency requirement"""
        vehicles = []
        for i in range(5):
            v_info = VehicleInfo(f"v{i}", (i*10, 0), 20, 0, {'cpu': 0.7})
            vehicles.append(v_info)
        
        start_time = time.time()
        template = self.manager.find_template_for_vehicles(vehicles)
        end_time = time.time()
        
        lookup_time = (end_time - start_time) * 1000  # Convert to milliseconds
        self.assertLess(lookup_time, 5.0, f"Template lookup took {lookup_time:.2f}ms, should be <5ms")
    
    def test_pipeline_recomposition_time(self):
        """Test pipeline recomposition <1.5s requirement"""
        from core.types import PipelineTemplate, TrainingConfig
        
        template = PipelineTemplate(
            template_id="test",
            resource_requirements=[ResourceClass.MEDIUM] * 3,
            expected_duration=15.0,
            communication_pattern=[(0, 1), (1, 2)],
            training_config=TrainingConfig()
        )
        
        vehicles = []
        for i in range(5):
            v_info = VehicleInfo(f"v{i}", (i*10, 0), 20, 0, {'cpu': 0.7})
            vehicles.append(v_info)
        
        start_time = time.time()
        pipeline_id = self.formation.initiate_pipeline_formation(template, vehicles)
        end_time = time.time()
        
        formation_time = end_time - start_time
        self.assertLess(formation_time, 1.5, f"Pipeline formation took {formation_time:.2f}s, should be <1.5s")

class TestIntegration(unittest.TestCase):
    """Integration tests for FHDP system components"""
    
    def setUp(self):
        self.system = FHDPSystem()
        self.system.start_system()
        
        # Create test edge server
        self.edge_server = EdgeServer()
        self.edge_server.start_server()
        
    def tearDown(self):
        self.system.stop_system()
        self.edge_server.stop_server()
    
    def test_end_to_end_flow(self):
        """Test end-to-end FHDP flow"""
        # Register vehicles
        vehicles = []
        for i in range(5):
            v_info = VehicleInfo(
                vehicle_id=f"vehicle_{i}",
                position=(i*50, 0),
                velocity=20.0,
                direction=0.0,
                resources={'cpu': 0.7, 'memory': 0.6, 'battery': 0.8}
            )
            vehicles.append(v_info)
            
            # Register with system
            self.system.register_vehicle(v_info)
            self.edge_server.register_vehicle(v_info)
        
        # Test pipeline formation
        candidates = self.system.participation_manager.get_pipeline_candidates(3)
        self.assertGreater(len(candidates), 0)
        
        # Test template matching
        candidate_vehicles = [self.system.registered_vehicles[vid] for vid in candidates[:3]]
        template = self.edge_server.find_pipeline_template([v.vehicle_id for v in candidate_vehicles])
        
        if template:
            self.assertIsInstance(template, PipelineTemplate)
        
        # Test system status
        status = self.system.get_system_status()
        self.assertEqual(status['registered_vehicles'], 5)
        self.assertGreater(status['uptime'], 0)

class TestScalability(unittest.TestCase):
    """Scalability tests for FHDP system"""
    
    def test_large_vehicle_population(self):
        """Test system with large number of vehicles"""
        system = FHDPSystem()
        system.start_system()
        
        try:
            # Register many vehicles
            num_vehicles = 50
            for i in range(num_vehicles):
                v_info = VehicleInfo(
                    vehicle_id=f"scale_vehicle_{i}",
                    position=(i*20, 0),
                    velocity=20.0,
                    direction=0.0,
                    resources={'cpu': 0.6, 'memory': 0.5, 'battery': 0.7}
                )
                system.register_vehicle(v_info)
            
            # Check system performance
            status = system.get_system_status()
            self.assertEqual(status['registered_vehicles'], num_vehicles)
            
            # Test that system remains responsive
            start_time = time.time()
            candidates = system.participation_manager.get_pipeline_candidates(10)
            end_time = time.time()
            
            response_time = end_time - start_time
            self.assertLess(response_time, 1.0, "System should remain responsive with many vehicles")
            
        finally:
            system.stop_system()

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMobilityPredictor,
        TestTemplateManager,
        TestResourceClassifier,
        TestPipelineFormation,
        TestTrainingExecutor,
        TestFHDPSystem,
        TestPerformanceRequirements,
        TestIntegration,
        TestScalability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)