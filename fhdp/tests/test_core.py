"""
Unit tests for FHDP core components
"""
import unittest
import torch
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.fhdp_system import SystemConfiguration, HybridParticipationManager
from core.types import VehicleInfo, ResourceClass, FairnessMetrics

class TestHybridParticipationManager(unittest.TestCase):
    """Tests for HybridParticipationManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SystemConfiguration()
        self.manager = HybridParticipationManager(self.config)
        
        # Create test vehicle info
        self.vehicle_info = VehicleInfo(
            vehicle_id="test_vehicle_1",
            position=(0.0, 0.0),
            velocity=10.0,
            direction=0.0,
            resources={
                'cpu': 0.8,
                'memory': 0.7,
                'battery': 0.9,
                'network_quality': 0.9,
                'thermal_state': 0.3
            }
        )
    
    def test_register_vehicle(self):
        """Test vehicle registration"""
        self.manager.register_vehicle(self.vehicle_info)
        self.assertIn(self.vehicle_info.vehicle_id, self.manager.active_vehicles)
    
    def test_unregister_vehicle(self):
        """Test vehicle unregistration"""
        self.manager.register_vehicle(self.vehicle_info)
        self.manager.unregister_vehicle(self.vehicle_info.vehicle_id)
        self.assertNotIn(self.vehicle_info.vehicle_id, self.manager.active_vehicles)
    
    def test_evaluate_participation_mode(self):
        """Test participation mode evaluation"""
        # Mock resource classifier and fairness metrics
        resource_class = ResourceClass.HIGH
        fairness_metrics = FairnessMetrics(
            vehicle_id=self.vehicle_info.vehicle_id,
            participation_count=0,
            last_participation=0.0,
            contribution_score=1.0,
            priority_weight=2.0
        )
        
        # Mock neighbors
        neighbors = {
            "neighbor_1": {},
            "neighbor_2": {},
            "neighbor_3": {}
        }
        
        # Test high resource vehicle with good fairness score
        mode = self.manager._evaluate_participation_mode(
            self.vehicle_info, resource_class, fairness_metrics, neighbors
        )
        self.assertEqual(mode.value, "pipeline")
    
    def test_get_pipeline_candidates(self):
        """Test pipeline candidate selection"""
        # Register multiple vehicles
        for i in range(10):
            vehicle = VehicleInfo(
                vehicle_id=f"test_vehicle_{i}",
                position=(i * 10.0, 0.0),
                velocity=10.0,
                direction=0.0,
                resources={
                    'cpu': 0.8,
                    'memory': 0.7,
                    'battery': 0.9,
                    'network_quality': 0.9,
                    'thermal_state': 0.3
                }
            )
            self.manager.register_vehicle(vehicle)
            # Mock participation decision for some vehicles
            if i % 2 == 0:
                self.manager.participation_decisions[vehicle.vehicle_id] = "pipeline"
        
        candidates = self.manager.get_pipeline_candidates(count=3)
        self.assertLessEqual(len(candidates), 3)

class TestMockAggregation(unittest.TestCase):
    """Tests for mock aggregation functionality"""
    
    def test_aggregation_result_structure(self):
        """Test that aggregation result has correct structure"""
        from core.fhdp_system import AsynchronousCoordinationManager
        
        config = SystemConfiguration()
        manager = AsynchronousCoordinationManager(config)
        
        # Create mock updates
        from core.types import ModelUpdate
        update1 = ModelUpdate(
            source_id="vehicle_1",
            update_data=torch.randn(100),
            metadata={"epoch": 1},
            training_mode="individual"
        )
        
        update2 = ModelUpdate(
            source_id="vehicle_2",
            update_data=torch.randn(100),
            metadata={"epoch": 1},
            training_mode="individual"
        )
        
        # Test mock aggregation
        result = manager._mock_aggregate_updates([update1, update2])
        self.assertIsInstance(result, dict)
        self.assertIn("layer1_weight", result)
        self.assertIn("layer1_bias", result)

if __name__ == '__main__':
    unittest.main()
