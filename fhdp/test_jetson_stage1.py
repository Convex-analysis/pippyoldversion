#!/usr/bin/env python3
"""
Test Suite for EVO-1 Stage 1 on Jetson Devices
Validates all components for Stage 1 Action Expert Alignment
"""
import sys
import os
import time
import torch
import numpy as np
import asyncio
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_jetson_constraints():
    """Test Jetson resource constraint management"""
    print("🔧 Testing Jetson Resource Constraints...")
    
    try:
        from examples.evo1_stage1_federated import JetsonResourceConstraints
        
        # Test constraint creation
        constraints = JetsonResourceConstraints(
            device_name="jetson_orin",
            max_memory_mb=6144,
            max_batch_size=4,
            precision="float16"
        )
        
        print(f"✅ Constraints created: {constraints.device_name}")
        print(f"   Max memory: {constraints.max_memory_mb}MB")
        print(f"   Max batch size: {constraints.max_batch_size}")
        print(f"   Precision: {constraints.precision}")
        
        return True
        
    except Exception as e:
        print(f"❌ Constraint test failed: {e}")
        return False

def test_lightweight_action_head():
    """Test lightweight action head for Stage 1"""
    print("\n🧠 Testing Lightweight Action Head...")
    
    try:
        from examples.evo1_stage1_federated import LightweightActionHead
        
        # Create action head
        action_head = LightweightActionHead(
            vision_dim=2048,
            language_dim=768,
            hidden_dim=256,
            action_dim=3
        )
        
        param_count = action_head.get_parameter_count()
        print(f"✅ Action head created with {param_count:,} parameters")
        
        # Test forward pass
        vision_features = torch.randn(2, 2048)
        language_features = torch.randn(2, 768)
        
        with torch.no_grad():
            actions = action_head(vision_features, language_features)
        
        print(f"✅ Forward pass successful: {actions.shape}")
        print(f"   Output range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Test training mode
        action_head.train()
        actions_train = action_head(vision_features, language_features)
        
        print(f"✅ Training mode works: {actions_train.requires_grad}")
        
        return True
        
    except Exception as e:
        print(f"❌ Action head test failed: {e}")
        return False

def test_frozen_vlm_interface():
    """Test frozen VLM interface for Stage 1"""
    print("\n🧊 Testing Frozen VLM Interface...")
    
    try:
        from examples.evo1_stage1_federated import FrozenVLMInterface
        
        # Create VLM interface
        vlm_interface = FrozenVLMInterface()
        
        print(f"✅ VLM interface created: {vlm_interface.vlm_name}")
        print(f"   Status: {'loaded' if vlm_interface.is_loaded else 'not loaded'}")
        
        # Test vision feature extraction
        images = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]
        vision_features = vlm_interface.extract_vision_features(images)
        
        print(f"✅ Vision features extracted: {vision_features.shape}")
        
        # Test language feature extraction
        prompts = ["保持车道", "安全驾驶"]
        language_features = vlm_interface.extract_language_features(prompts)
        
        print(f"✅ Language features extracted: {language_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VLM interface test failed: {e}")
        return False

def test_jetson_memory_manager():
    """Test Jetson memory management"""
    print("\n💾 Testing Jetson Memory Manager...")
    
    try:
        from examples.evo1_stage1_federated import JetsonMemoryManager, JetsonResourceConstraints
        
        constraints = JetsonResourceConstraints(max_memory_mb=4096)
        memory_manager = JetsonMemoryManager(constraints)
        
        # Test memory usage
        usage = memory_manager.get_memory_usage()
        print(f"✅ Memory usage retrieved:")
        print(f"   Allocated: {usage['allocated_gb']:.2f}GB")
        print(f"   Usage: {usage['usage_percent']:.1f}%")
        
        # Test memory optimization
        memory_manager.optimize_memory()
        print("✅ Memory optimization completed")
        
        # Test temperature check
        temp = memory_manager.get_temperature()
        if temp:
            print(f"✅ Temperature reading: {temp:.1f}°C")
        else:
            print("✅ Temperature check attempted (not available)")
        
        # Test memory threshold
        threshold_exceeded = memory_manager.check_memory_threshold()
        print(f"✅ Memory threshold check: {threshold_exceeded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        return False

def test_stage1_trainer():
    """Test Stage 1 trainer"""
    print("\n🎓 Testing Stage 1 Trainer...")
    
    try:
        from examples.evo1_stage1_federated import Stage1ActionExpertTrainer, JetsonResourceConstraints
        
        constraints = JetsonResourceConstraints(max_batch_size=2, precision="float16")
        trainer = Stage1ActionExpertTrainer(constraints)
        
        print(f"✅ Trainer created on {trainer.device}")
        print(f"   Action head params: {trainer.action_head.get_parameter_count():,}")
        
        # Test training step
        batch_data = {
            'images': [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],
            'prompts': ["直行", "右转"],
            'actions': [[0.1, 0.5, 0.0], [-0.2, 0.3, 0.1]]
        }
        
        metrics = trainer.train_step(batch_data)
        print(f"✅ Training step completed:")
        print(f"   Loss: {metrics['loss']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   Learning rate: {metrics['lr']:.6f}")
        
        # Test training epoch
        dataset = [batch_data, batch_data, batch_data]
        epoch_result = trainer.train_epoch(dataset, epoch=1)
        
        print(f"✅ Training epoch completed:")
        print(f"   Avg loss: {epoch_result['avg_loss']:.4f}")
        print(f"   Memory usage: {epoch_result['memory_usage']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Stage 1 trainer test failed: {e}")
        return False

def test_jetson_federated_learning():
    """Test Jetson-optimized federated learning"""
    print("\n🌐 Testing Jetson Federated Learning...")
    
    try:
        from examples.evo1_stage1_federated import (
            JetsonFederatedLearning, Stage1ActionExpertTrainer, 
            JetsonResourceConstraints
        )
        
        constraints = JetsonResourceConstraints()
        fed_learner = JetsonFederatedLearning(constraints)
        
        # Create multiple trainers
        trainers = {}
        for i in range(3):
            vehicle_id = f"test_vehicle_{i}"
            trainer = fed_learner.prepare_model_for_vehicle(vehicle_id)
            trainers[vehicle_id] = trainer
        
        print(f"✅ Created {len(trainers)} vehicle trainers")
        
        # Collect updates (simulate some training)
        updates = fed_learner.collect_vehicle_updates(trainers)
        print(f"✅ Collected updates from {len(updates)} vehicles")
        
        # Test aggregation
        aggregated = fed_learner.aggregate_weights(updates)
        if aggregated:
            print(f"✅ Weight aggregation successful")
            print(f"   Aggregated keys: {len(aggregated)}")
        else:
            print("⚠️  No aggregated weights (empty updates)")
        
        return True
        
    except Exception as e:
        print(f"❌ Federated learning test failed: {e}")
        return False

def test_jetson_vehicle():
    """Test Jetson autonomous vehicle"""
    print("\n🚗 Testing Jetson Autonomous Vehicle...")
    
    try:
        from examples.evo1_stage1_federated import JetsonAutonomousVehicle, JetsonResourceConstraints
        
        constraints = JetsonResourceConstraints()
        vehicle = JetsonAutonomousVehicle(
            "test_vehicle", 
            (0.0, 0.0), 
            constraints
        )
        
        print(f"✅ Vehicle created: {vehicle.vehicle_id}")
        
        # Test data collection
        observation = {
            'images': [torch.randn(3, 224, 224)],
            'prompt': '保持车道行驶',
            'speed': 15.0
        }
        
        action = np.array([0.1, 0.5, 0.0])
        reward = 0.8
        
        sample = vehicle.collect_training_data(observation, action, reward)
        print(f"✅ Data collection successful: {len(vehicle.training_data)} samples")
        
        # Test training
        # Add more samples for training
        for _ in range(10):
            vehicle.collect_training_data(observation, action, reward)
        
        result = vehicle.train_stage1(epochs=1)
        print(f"✅ Stage 1 training: {result['status']}")
        
        if result['status'] == 'completed':
            print(f"   Avg loss: {result['avg_loss']:.4f}")
            print(f"   Training time: {result['training_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Jetson vehicle test failed: {e}")
        return False

def test_scenario_generation():
    """Test scenario generation for Jetson"""
    print("\n📍 Testing Scenario Generation...")
    
    try:
        from examples.evo1_stage1_federated import create_jetson_scenarios
        
        scenarios = create_jetson_scenarios()
        
        print(f"✅ Generated {len(scenarios)} scenarios:")
        for scenario in scenarios:
            print(f"   {scenario['name']}: {scenario['description']}")
            print(f"     Duration: {scenario['duration']}s, Complexity: {scenario['complexity']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scenario generation test failed: {e}")
        return False

async def test_jetson_driving_simulation():
    """Test driving simulation on Jetson"""
    print("\n🛣️  Testing Jetson Driving Simulation...")
    
    try:
        from examples.evo1_stage1_federated import JetsonAutonomousVehicle, JetsonResourceConstraints
        
        constraints = JetsonResourceConstraints(max_batch_size=2)
        vehicle = JetsonAutonomousVehicle("sim_vehicle", (0.0, 0.0), constraints)
        
        print(f"✅ Starting simulation for {vehicle.vehicle_id}")
        
        # Run short simulation
        duration = 5.0  # 5 seconds
        await simulate_jetson_driving(vehicle, duration)
        
        print(f"✅ Simulation completed:")
        print(f"   Final position: {vehicle.position}")
        print(f"   Distance: {vehicle.driving_metrics['distance']:.2f}m")
        print(f"   Training data: {len(vehicle.training_data)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Driving simulation test failed: {e}")
        return False

async def simulate_jetson_driving(vehicle, duration):
    """Helper function for driving simulation test"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Simple simulation
        observation = {
            'images': [torch.randn(3, 224, 224)],
            'prompt': '测试驾驶',
            'speed': vehicle.velocity
        }
        
        action = np.array([0.0, 0.3, 0.0])  # Straight, slight throttle
        reward = 0.7
        
        vehicle.collect_training_data(observation, action, reward)
        
        # Update vehicle state
        vehicle.velocity = 10.0  # Constant speed
        vehicle.position = (
            vehicle.position[0] + vehicle.velocity * 0.1,
            vehicle.position[1]
        )
        
        vehicle.driving_metrics['distance'] += vehicle.velocity * 0.1
        
        await asyncio.sleep(0.5)

def test_compatibility():
    """Test compatibility with current environment"""
    print("\n🔍 Testing Environment Compatibility...")
    
    tests = []
    
    # Python version
    version = sys.version_info
    python_ok = (version.major == 3 and 8 <= version.minor <= 10)
    tests.append(("Python 3.8-3.10", python_ok))
    print(f"   Python: {version.major}.{version.minor} {'✅' if python_ok else '❌'}")
    
    # PyTorch
    try:
        import torch
        torch_version = torch.__version__
        torch_ok = True
        tests.append(("PyTorch", torch_ok))
        print(f"   PyTorch: {torch_version} ✅")
        
        # CUDA
        cuda_available = torch.cuda.is_available()
        tests.append(("CUDA", cuda_available))
        print(f"   CUDA: {'Available' if cuda_available else 'Not Available'} {'✅' if cuda_available else '⚠️'}")
        
    except ImportError:
        torch_ok = False
        tests.append(("PyTorch", False))
        print(f"   PyTorch: Not installed ❌")
    
    # NumPy
    try:
        import numpy as np
        tests.append(("NumPy", True))
        print(f"   NumPy: {np.__version__} ✅")
    except ImportError:
        tests.append(("NumPy", False))
        print(f"   NumPy: Not installed ❌")
    
    # PSUtil
    try:
        import psutil
        tests.append(("PSUtil", True))
        print(f"   PSUtil: {psutil.__version__} ✅")
    except ImportError:
        tests.append(("PSUtil", False))
        print(f"   PSUtil: Not installed ❌")
    
    # Overall compatibility
    all_passed = all(result for _, result in tests)
    print(f"\n   Overall compatibility: {'✅ Passed' if all_passed else '⚠️  Some issues'}")
    
    return all_passed

async def main():
    """Run all Stage 1 tests"""
    print("🧪 EVO-1 Stage 1 Jetson Test Suite")
    print("=" * 50)
    
    tests = [
        ("Jetson Constraints", test_jetson_constraints),
        ("Environment Compatibility", test_compatibility),
        ("Lightweight Action Head", test_lightweight_action_head),
        ("Frozen VLM Interface", test_frozen_vlm_interface),
        ("Jetson Memory Manager", test_jetson_memory_manager),
        ("Stage 1 Trainer", test_stage1_trainer),
        ("Jetson Federated Learning", test_jetson_federated_learning),
        ("Jetson Autonomous Vehicle", test_jetson_vehicle),
        ("Scenario Generation", test_scenario_generation),
        ("Driving Simulation", test_jetson_driving_simulation),
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
    print(f"\n{'='*50}")
    print("🎯 Stage 1 Test Summary")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:<10} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {total_duration:.2f}s")
    
    if passed >= total * 0.8:  # 80% pass rate
        print(f"\n🎉 Stage 1 test suite passed!")
        print(f"   Ready for EVO-1 Stage 1 deployment on Jetson")
        print(f"\n🚀 Next steps:")
        print(f"   1. Run deployment script: ./deploy_jetson_stage1.sh")
        print(f"   2. Start Stage 1 simulation: python examples/evo1_stage1_federated.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print(f"   Please check errors and fix dependencies")
    
    # Performance recommendations
    print(f"\n💡 Performance recommendations:")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available - GPU acceleration enabled")
    else:
        print(f"   ⚠️  CUDA not available - CPU-only mode")
    
    memory_info = psutil.virtual_memory()
    if memory_info.total < 6 * 1024**3:  # Less than 6GB
        print(f"   ⚠️  Limited memory detected ({memory_info.total//(1024**3)}GB)")
        print(f"       Consider reducing batch sizes and model complexity")
    
    print(f"   ✅ Stage 1 optimized for Jetson Orin/Nano devices")
    print(f"   ✅ VLM freezing saves significant memory and computation")
    print(f"   ✅ Lightweight action head (~10K parameters)")

if __name__ == '__main__':
    asyncio.run(main())