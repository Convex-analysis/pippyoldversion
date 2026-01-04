#!/usr/bin/env python3
"""
Real Training Simulation for FHDP System

This script implements actual neural network training with real datasets
to demonstrate FHDP's distributed training capabilities.
"""
import sys
import os
import time
import random
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fhdp.core import FHDPSystem, SystemConfiguration
from fhdp.edge_server import EdgeServer
from fhdp.vehicle_layer import Vehicle
from fhdp.core.types import VehicleInfo, TrainingConfig, TrainingMode
from fhdp.vehicle_layer.training_engine import TrainingTask, TrainingExecutor

class SimpleCNN(nn.Module):
    """Simple CNN for vehicle image classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class VehicleDatasetManager:
    """Manages datasets for different vehicles"""
    
    def __init__(self, dataset_name="MNIST", num_classes=10):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.full_dataset = None
        self.vehicle_datasets = {}
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        """Prepare the base dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if self.dataset_name == "MNIST":
            self.full_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
        elif self.dataset_name == "CIFAR10":
            self.full_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def get_vehicle_dataset(self, vehicle_id: str, num_samples: int = 1000) -> DataLoader:
        """Get dataset subset for a specific vehicle"""
        if vehicle_id not in self.vehicle_datasets:
            # Randomly sample data for this vehicle (simulating private data)
            total_samples = len(self.full_dataset)
            indices = random.sample(range(total_samples), min(num_samples, total_samples))
            
            # Create subset
            subset = Subset(self.full_dataset, indices)
            
            # Create dataloader
            self.vehicle_datasets[vehicle_id] = DataLoader(
                subset, batch_size=32, shuffle=True, num_workers=2
            )
        
        return self.vehicle_datasets[vehicle_id]

class RealTrainingSimulation:
    """Real training simulation with actual neural networks"""
    
    def __init__(self, num_vehicles=6):
        self.num_vehicles = num_vehicles
        self.dataset_manager = VehicleDatasetManager("MNIST")
        self.global_model = SimpleCNN(num_classes=10)
        self.vehicle_models = {}
        self.training_results = {}
        
    def create_real_vehicles(self, num_vehicles: int) -> list:
        """Create vehicles with realistic configurations"""
        vehicles = []
        
        for i in range(num_vehicles):
            # Random position along highway
            x = random.uniform(-500, 500)
            y = random.uniform(-50, 50)
            
            # Random velocity (10-30 m/s = 36-108 km/h)
            velocity = random.uniform(10, 30)
            
            # Random direction (mostly forward with some variation)
            direction = random.uniform(-0.2, 0.2)
            
            # Simulate different vehicle capabilities
            vehicle_types = {
                'high_power': {'cpu': 0.8, 'memory': 0.7, 'battery': 0.9},
                'medium_power': {'cpu': 0.6, 'memory': 0.5, 'battery': 0.7},
                'low_power': {'cpu': 0.4, 'memory': 0.3, 'battery': 0.5}
            }
            
            vehicle_type = random.choice(list(vehicle_types.keys()))
            resources = vehicle_types[vehicle_type].copy()
            resources.update({
                'network_quality': random.uniform(0.6, 1.0),
                'thermal_state': random.uniform(0.1, 0.3)
            })
            
            # Set training capability based on vehicle type
            training_capability = {
                'high_power': 0.9,
                'medium_power': 0.6,
                'low_power': 0.3
            }[vehicle_type]
            
            vehicle_info = VehicleInfo(
                vehicle_id=f"vehicle_{i:03d}_{vehicle_type}",
                position=(x, y),
                velocity=velocity,
                direction=direction,
                resources=resources,
                training_capability=training_capability
            )
            
            vehicles.append(vehicle_info)
            
            # Create model for this vehicle
            self.vehicle_models[vehicle_info.vehicle_id] = SimpleCNN(num_classes=10)
        
        return vehicles
    
    def train_vehicle_locally(self, vehicle_id: str, epochs: int = 2) -> dict:
        """Train model locally on vehicle data"""
        print(f"🚗 Starting real training for {vehicle_id}...")
        
        # Get vehicle's private dataset
        train_loader = self.dataset_manager.get_vehicle_dataset(vehicle_id, num_samples=1000)
        
        # Get vehicle's model
        model = self.vehicle_models[vehicle_id]
        
        # Setup training
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += target.size(0)
                
                if batch_idx % 10 == 0:
                    print(f"  {vehicle_id} Epoch {epoch+1}/{epochs}, "
                          f"Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
            
            # Epoch statistics
            epoch_accuracy = 100. * epoch_correct / epoch_total
            print(f"  {vehicle_id} Epoch {epoch+1} completed, "
                  f"Loss: {epoch_loss/len(train_loader):.4f}, "
                  f"Accuracy: {epoch_accuracy:.2f}%")
        
        training_time = time.time() - start_time
        
        # Calculate model update (parameter differences from global model)
        model_update = self._calculate_model_update(model, self.global_model)
        
        # Store training results
        result = {
            'vehicle_id': vehicle_id,
            'epochs': epochs,
            'training_time': training_time,
            'final_loss': total_loss / (epochs * len(train_loader)),
            'accuracy': 100. * correct / total,
            'samples_processed': total,
            'model_update': model_update,
            'model_size': sum(p.numel() for p in model.parameters())
        }
        
        self.training_results[vehicle_id] = result
        
        print(f"✅ {vehicle_id} training completed in {training_time:.2f}s, "
              f"Accuracy: {result['accuracy']:.2f}%")
        
        return result
    
    def _calculate_model_update(self, local_model: nn.Module, global_model: nn.Module) -> dict:
        """Calculate parameter differences between local and global model"""
        update = {}
        local_params = dict(local_model.named_parameters())
        global_params = dict(global_model.named_parameters())
        
        for name in local_params:
            if name in global_params:
                # Calculate difference
                param_diff = local_params[name].data - global_params[name].data
                update[name] = param_diff.detach().clone()
        
        return update
    
    def aggregate_model_updates(self, vehicle_results: list) -> nn.Module:
        """Aggregate model updates from multiple vehicles"""
        print(f"🔄 Aggregating updates from {len(vehicle_results)} vehicles...")
        
        # Simple averaging aggregation
        aggregated_params = {}
        
        # Get the first available vehicle model to get parameter names
        first_vehicle_id = list(self.vehicle_models.keys())[0]
        for name in self.vehicle_models[first_vehicle_id].state_dict():
            # Initialize with zeros
            aggregated_params[name] = torch.zeros_like(
                self.global_model.state_dict()[name]
            )
        
        # Sum updates
        total_weight = 0
        for result in vehicle_results:
            weight = result['samples_processed']  # Weight by data size
            total_weight += weight
            
            for name, param_diff in result['model_update'].items():
                aggregated_params[name] += param_diff * weight
        
        # Average updates
        for name in aggregated_params:
            aggregated_params[name] /= total_weight
        
        # Apply updates to global model
        global_state = self.global_model.state_dict()
        for name in aggregated_params:
            global_state[name] += aggregated_params[name]
        
        self.global_model.load_state_dict(global_state)
        
        print("✅ Model aggregation completed")
        return self.global_model
    
    def evaluate_global_model(self) -> float:
        """Evaluate global model on test set"""
        # Load test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        print(f"🎯 Global model accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def run_real_training_simulation(self, num_rounds: int = 3):
        """Run real federated learning simulation"""
        print("=== FHDP Real Training Simulation ===")
        print("🧠 Training actual neural networks on MNIST dataset")
        print()
        
        # Create vehicles
        vehicles = self.create_real_vehicles(self.num_vehicles)
        print(f"📊 Created {len(vehicles)} vehicles with heterogeneous capabilities")
        
        for vehicle in vehicles:
            # Extract vehicle type from ID for display
            # ID format: "vehicle_{i:03d}_{vehicle_type}"
            vehicle_parts = vehicle.vehicle_id.split('_')
            vehicle_type = '_'.join(vehicle_parts[2:])  # Join all parts after "vehicle_{number}"
            
            gpu_status = {
                'high_power': 'GPU=Yes',
                'medium_power': 'GPU=Yes', 
                'low_power': 'GPU=No'
            }.get(vehicle_type, 'GPU=Unknown')
            
            batch_size = {
                'high_power': '64',
                'medium_power': '32',
                'low_power': '16'
            }.get(vehicle_type, '32')
            
            memory = {
                'high_power': '4096MB',
                'medium_power': '2048MB',
                'low_power': '1024MB'
            }.get(vehicle_type, '1024MB')
            
            print(f"  {vehicle.vehicle_id}: {gpu_status}, BatchSize={batch_size}, Memory={memory}")
        print()
        
        # Evaluate initial global model
        print("📈 Evaluating initial global model...")
        initial_accuracy = self.evaluate_global_model()
        
        # Training rounds
        global_accuracies = [initial_accuracy]
        
        for round_num in range(num_rounds):
            print(f"\n🚀 === Training Round {round_num + 1}/{num_rounds} ===")
            
            # Distribute global model to vehicles
            for vehicle_id in self.vehicle_models:
                self.vehicle_models[vehicle_id].load_state_dict(self.global_model.state_dict())
            
            # Parallel training on all vehicles
            training_threads = []
            round_results = []
            
            for vehicle in vehicles:
                thread = threading.Thread(
                    target=lambda v=vehicle: round_results.append(
                        self.train_vehicle_locally(v.vehicle_id, epochs=2)
                    )
                )
                thread.start()
                training_threads.append(thread)
            
            # Wait for all training to complete
            for thread in training_threads:
                thread.join()
            
            # Aggregate model updates
            self.aggregate_model_updates(round_results)
            
            # Evaluate updated global model
            round_accuracy = self.evaluate_global_model()
            global_accuracies.append(round_accuracy)
            
            # Print round statistics
            avg_accuracy = np.mean([r['accuracy'] for r in round_results])
            total_samples = sum([r['samples_processed'] for r in round_results])
            total_time = sum([r['training_time'] for r in round_results])
            
            print(f"📊 Round {round_num + 1} Statistics:")
            print(f"  Average vehicle accuracy: {avg_accuracy:.2f}%")
            print(f"  Total samples processed: {total_samples}")
            print(f"  Total training time: {total_time:.2f}s")
            print(f"  Global model accuracy: {round_accuracy:.2f}%")
            print(f"  Accuracy improvement: {round_accuracy - global_accuracies[-2]:.2f}%")
        
        # Final results
        print(f"\n🎉 === Final Results ===")
        print(f"Initial global accuracy: {initial_accuracy:.2f}%")
        print(f"Final global accuracy: {global_accuracies[-1]:.2f}%")
        print(f"Total improvement: {global_accuracies[-1] - initial_accuracy:.2f}%")
        
        # Individual vehicle statistics
        print(f"\n📈 Vehicle Performance Summary:")
        for vehicle_id, result in self.training_results.items():
            print(f"  {vehicle_id}:")
            print(f"    Final accuracy: {result['accuracy']:.2f}%")
            print(f"    Training time: {result['training_time']:.2f}s")
            print(f"    Model size: {result['model_size']:,} parameters")

def main():
    """Run real training simulation"""
    print("Starting FHDP Real Training Simulation...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create simulation
    sim = RealTrainingSimulation(num_vehicles=6)
    
    # Run simulation
    sim.run_real_training_simulation(num_rounds=3)
    
    print("\n✅ Real training simulation completed successfully!")

if __name__ == '__main__':
    main()