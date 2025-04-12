import pickle
import torch
from timm.models import create_model

# Create the model
model = create_model("memfuser_baseline_e1d3", pretrained=False)

# Try to pickle it
try:
    print("Attempting to pickle the model...")
    pickle_data = pickle.dumps(model)
    print(f"Successfully pickled model! Size: {len(pickle_data) / (1024 * 1024):.2f} MB")
    
    # Try to unpickle it
    print("Attempting to unpickle the model...")
    unpickled_model = pickle.loads(pickle_data)
    print("Successfully unpickled model!")
    
    # Verify that the unpickled model has the same structure
    print(f"Original model type: {type(model)}")
    print(f"Unpickled model type: {type(unpickled_model)}")
    
    # Check if the model can be used for inference
    print("Testing model inference...")
    with torch.no_grad():
        # Create dummy inputs
        dummy_inputs = {
            "rgb_front": torch.randn(1, 3, 224, 224),
            "rgb_left": torch.randn(1, 3, 224, 224),
            "rgb_right": torch.randn(1, 3, 224, 224),
            "rgb_rear": torch.randn(1, 3, 224, 224),
            "rgb_center": torch.randn(1, 3, 224, 224),
            "lidar": torch.randn(1, 9, 10000),
            "num_points": torch.tensor([10000]),
            "velocity": torch.randn(1, 1)
        }
        
        # Run inference with the original model
        model.eval()
        original_output = model(dummy_inputs)
        
        # Run inference with the unpickled model
        unpickled_model.eval()
        unpickled_output = unpickled_model(dummy_inputs)
        
        print("Both models successfully ran inference!")
        
except Exception as e:
    print(f"Error: {e}")
