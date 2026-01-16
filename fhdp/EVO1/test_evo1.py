#!/usr/bin/env python3
"""
Simple test script for EVO-1 model functionality
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_evo1_model():
    """Test EVO-1 model basic functionality"""
    print("🚀 Testing EVO-1 Autonomous Driving Model")
    print("=" * 60)
    
    try:
        # Import model
        from model.evo1_driving import EVO1Driving, ModelConfig
        print("✅ Model imported successfully")
        
        # Create configuration
        config = ModelConfig()
        config.vision_encoder = "OpenGVLab/InternVL3-1B"
        config.sequence_length = 32
        print("✅ Configuration created")
        
        # Create model
        model = EVO1Driving(config)
        print(f"✅ Model created successfully")
        
        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Model Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 2
        sequence_length = 32
        
        # Create dummy inputs
        images = torch.randn(batch_size, 3, 224, 224)
        text_inputs = torch.randint(0, 1000, (batch_size, sequence_length))
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(images, text_inputs)
            print("✅ Forward pass successful")
            
        # Print output shapes
        print(f"📤 Output shapes:")
        if hasattr(outputs, 'actions'):
            print(f"   Actions: {outputs.actions.shape}")
        if hasattr(outputs, 'waypoints'):
            print(f"   Waypoints: {outputs.waypoints.shape}")
        if hasattr(outputs, 'uncertainty'):
            print(f"   Uncertainty: {outputs.uncertainty.shape}")
        
        # Test gradient computation
        model.train()
        dummy_targets = torch.randn_like(outputs.actions if hasattr(outputs, 'actions') else outputs.waypoints)
        loss = torch.nn.functional.mse_loss(outputs.actions if hasattr(outputs, 'actions') else outputs.waypoints, dummy_targets)
        loss.backward()
        
        print(f"✅ Gradient computation successful (loss: {loss.item():.6f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_federated_evo1():
    """Test Federated EVO-1 variant"""
    print("\n🔗 Testing Federated EVO-1 Model")
    print("=" * 60)
    
    try:
        from model.evo1_driving import FederatedEVO1Driving, ModelConfig
        
        # Create configuration
        config = ModelConfig()
        
        # Create federated model
        fed_model = FederatedEVO1Driving(config, num_clients=4)
        print("✅ Federated model created successfully")
        
        # Test federated aggregation
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        text_inputs = torch.randint(0, 1000, (batch_size, 32))
        
        # Run inference
        fed_model.eval()
        with torch.no_grad():
            outputs = fed_model(images, text_inputs)
            print("✅ Federated forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Federated test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 EVO-1 Model Test Suite")
    print("Testing core EVO-1 autonomous driving functionality\n")
    
    # Test basic model
    basic_test_passed = test_evo1_model()
    
    # Test federated model
    fed_test_passed = test_federated_evo1()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Basic EVO-1: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"   Federated EVO-1: {'✅ PASSED' if fed_test_passed else '❌ FAILED'}")
    
    if basic_test_passed and fed_test_passed:
        print("\n🎉 All tests passed! EVO-1 model is working correctly.")
        print("Ready for training and inference.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return basic_test_passed and fed_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)