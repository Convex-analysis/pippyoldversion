#!/usr/bin/env python3
"""
Test script for EVO-1 + FHDP Autonomous Driving Integration

This script tests individual components before running the full simulation.
"""
import sys
import os
import asyncio
import time
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Core FHDP imports
        from fhdp.core import FHDPSystem, SystemConfiguration
        from fhdp.edge_server import EdgeServer
        from fhdp.vehicle_layer import Vehicle
        from fhdp.core.types import VehicleInfo, TrainingConfig, TrainingMode
        print("✅ FHDP imports successful")
        
        # ML imports
        import torch
        import torchvision
        import cv2
        import numpy as np
        print("✅ ML imports successful")
        
        # WebSocket imports
        import websockets
        import json
        print("✅ WebSocket imports successful")
        
        # Custom imports
        from examples.autonomous_driving_simulation import (
            EVO1Observation, EVO1ModelClient, NuScenesDataLoader,
            AutonomousDrivingVehicle, FederatedEVO1Trainer
        )
        print("✅ Custom autonomous driving imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_evo1_observation():
    """Test EVO-1 observation creation"""
    print("\n🧠 Testing EVO-1 observation...")
    
    try:
        obs = EVO1Observation()
        
        # Test camera images
        obs.images = [np.random.randint(0, 255, (480, 640, 3)) for _ in range(6)]
        obs.image_masks = [1] * 6
        
        # Test vehicle state
        obs.state = [20.0, 0.1, 1.0, 0.05]  # speed, yaw, accel, steering
        obs.action_mask = [[1, 1, 1]]
        obs.prompt = "保持车道并平稳行驶"
        
        print(f"✅ Observation created: {len(obs.images)} cameras, state: {obs.state}")
        return True
        
    except Exception as e:
        print(f"❌ EVO-1 observation test failed: {e}")
        return False

def test_nuscenes_dataloader():
    """Test nuScenes data loader"""
    print("\n📁 Testing nuScenes data loader...")
    
    try:
        data_loader = NuScenesDataLoader()
        
        # Test scene loading
        data_loader.load_scene(0)
        print(f"✅ Scene loaded: {data_loader.current_scene['name']}")
        
        # Test frame generation
        frame = data_loader.get_next_frame()
        if frame:
            print(f"✅ Frame generated: {len(frame.images)} cameras, prompt: '{frame.prompt}'")
        else:
            print("⚠️  No frame generated (might be expected)")
        
        return True
        
    except Exception as e:
        print(f"❌ NuScenes data loader test failed: {e}")
        return False

async def test_evo1_client():
    """Test EVO-1 model client"""
    print("\n🤖 Testing EVO-1 client...")
    
    try:
        # Create client (will use fallback if server not running)
        client = EVO1ModelClient("ws://localhost:8765")
        
        # Try to connect (will fail gracefully if server not running)
        await client.connect()
        
        # Create test observation
        obs = EVO1Observation()
        obs.images = [np.random.randint(0, 255, (480, 640, 3)) for _ in range(6)]
        obs.image_masks = [1] * 6
        obs.state = [15.0, 0.0, 0.0, 0.0]
        obs.action_mask = [[1, 1, 1]]
        obs.prompt = "直行"
        
        # Test action prediction
        action = await client.predict_action(obs)
        print(f"✅ Action predicted: {action} (shape: {action.shape})")
        
        await client.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ EVO-1 client test failed: {e}")
        return False

def test_federated_trainer():
    """Test federated EVO-1 trainer"""
    print("\n🎓 Testing federated trainer...")
    
    try:
        trainer = FederatedEVO1Trainer()
        
        # Test model preparation
        trainer.prepare_vehicle_model("test_vehicle")
        print("✅ Vehicle model prepared")
        
        # Test training (with mock data)
        observations = [EVO1Observation() for _ in range(10)]
        actions = [np.random.randn(3) for _ in range(10)]
        
        for obs in observations:
            obs.images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(6)]
            obs.image_masks = [1] * 6
            obs.state = [10.0, 0.0, 0.0, 0.0]
            obs.action_mask = [[1, 1, 1]]
            obs.prompt = "测试"
        
        stats = trainer.train_vehicle_model("test_vehicle", observations, actions)
        print(f"✅ Training completed: {stats['training_time']:.2f}s, loss: {stats['final_loss']:.4f}")
        
        # Test aggregation
        trainer.aggregate_model_updates(["test_vehicle"])
        print("✅ Model aggregation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Federated trainer test failed: {e}")
        return False

def test_autonomous_vehicle():
    """Test autonomous driving vehicle"""
    print("\n🚗 Testing autonomous vehicle...")
    
    try:
        # Create components
        evo1_client = EVO1ModelClient()
        data_loader = NuScenesDataLoader()
        data_loader.load_scene(0)
        
        # Create vehicle
        vehicle = AutonomousDrivingVehicle(
            vehicle_id="test_vehicle",
            initial_position=(0.0, 0.0),
            evo1_client=evo1_client,
            data_loader=data_loader
        )
        print("✅ Autonomous vehicle created")
        
        # Test action application
        action = np.array([0.1, 0.5, 0.0])  # steering, throttle, brake
        vehicle._apply_action(action)
        print(f"✅ Action applied: position={vehicle.position}, velocity={vehicle.velocity}")
        
        # Test metrics
        vehicle._update_metrics(action, EVO1Observation())
        print(f"✅ Metrics updated: comfort={vehicle.comfort_score:.1f}, efficiency={vehicle.efficiency_score:.1f}")
        
        # Test report generation
        report = vehicle.get_driving_report()
        print(f"✅ Report generated: {report['vehicle_id']}, distance={report['total_distance']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Autonomous vehicle test failed: {e}")
        return False

def test_fhdp_integration():
    """Test FHDP system integration"""
    print("\n🔧 Testing FHDP integration...")
    
    try:
        # Create FHDP configuration
        config = SystemConfiguration(
            max_vehicles_per_region=2,
            pipeline_formation_interval=5.0,
            model_broadcast_interval=10.0,
            enable_pipeline_training=True,
            enable_individual_training=True,
            fairness_enabled=True
        )
        
        # Create system
        system = FHDPSystem(config)
        system.start_system()
        print("✅ FHDP system started")
        
        # Create edge server
        edge_server = EdgeServer()
        edge_server.start_server()
        print("✅ Edge server started")
        
        # Register a test vehicle
        vehicle_info = VehicleInfo(
            vehicle_id="test_vehicle",
            position=(0.0, 0.0),
            velocity=15.0,
            direction=0.0,
            resources={'cpu': 0.7, 'memory': 0.6, 'battery': 0.8},
            training_capability=0.7
        )
        
        system.register_vehicle(vehicle_info)
        edge_server.register_vehicle(vehicle_info)
        print("✅ Vehicle registered")
        
        # Check system status
        system_status = system.get_system_status()
        server_status = edge_server.get_server_statistics()
        
        print(f"✅ System status: {system_status['registered_vehicles']} vehicles")
        print(f"✅ Server status: {server_status['connected_vehicles']} connected")
        
        # Cleanup
        system.stop_system()
        edge_server.stop_server()
        print("✅ System stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ FHDP integration test failed: {e}")
        return False

def test_torch_gpu():
    """Test PyTorch GPU availability"""
    print("\n🔥 Testing PyTorch GPU...")
    
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            
            # Test GPU tensor operation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✅ GPU tensor operation successful")
        else:
            print("⚠️  GPU not available, using CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Running EVO-1 + FHDP Autonomous Driving Tests\n")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("PyTorch GPU Test", test_torch_gpu),
        ("EVO-1 Observation", test_evo1_observation),
        ("NuScenes Data Loader", test_nuscenes_dataloader),
        ("EVO-1 Client", test_evo1_client),
        ("Federated Trainer", test_federated_trainer),
        ("Autonomous Vehicle", test_autonomous_vehicle),
        ("FHDP Integration", test_fhdp_integration),
    ]
    
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        duration = time.time() - start_time
        print(f"⏱️  Test completed in {duration:.2f}s")
    
    total_duration = time.time() - total_start
    print(f"\n{'='*60}")
    print("🎯 Test Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:<10} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {total_duration:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for autonomous driving simulation!")
        print("\nNext steps:")
        print("1. Run: python examples/autonomous_driving_simulation.py")
        print("2. Or use: ./run_autonomous_driving.sh")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements_evo1.txt")
        print("2. Check Python version: python --version (should be 3.10)")
        print("3. Verify dataset setup if needed")
        print("4. Check EVO-1 server if running")

if __name__ == '__main__':
    asyncio.run(main())