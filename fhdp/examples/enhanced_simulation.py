#!/usr/bin/env python3
"""
Enhanced Simple Simulation with Real Training Integration

This version adds real neural network training to the original simple simulation,
maintaining the same interface but with actual model training.
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

class LightweightCNN(nn.Module):
    """Lightweight CNN suitable for vehicle training"""
    
    def __init__(self, num_classes=10):
        super(LightweightCNN, self).__init__()
        
        # Simple architecture for quick training
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Flatten size: 32 * 7 * 7 = 1568
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_parameter_count(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

class RealTrainingEngine:
    """Real training engine that works with FHDP vehicles"""
    
    def __init__(self):
        self.global_model = None
        self.vehicle_models = {}
        self.dataset_cache = {}
        self.training_stats = {}
        
    def prepare_global_model(self, model_path: str = None):
        """Prepare the global model"""
        if model_path and os.path.exists(model_path):
            self.global_model = torch.load(model_path)
        else:
            self.global_model = LightweightCNN(num_classes=10)
        
        print(f"🧠 Global model created with {self.global_model.get_parameter_count():,} parameters")
        
    def get_vehicle_dataset(self, vehicle_id: str, num_samples: int = 500):
        """Get private dataset for vehicle"""
        if vehicle_id not in self.dataset_cache:
            # Download MNIST if not exists
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            full_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            
            # Randomly sample data for this vehicle (simulating private data)
            total_samples = len(full_dataset)
            indices = random.sample(range(total_samples), min(num_samples, total_samples))
            
            # Create subset and dataloader
            subset = Subset(full_dataset, indices)
            dataloader = DataLoader(subset, batch_size=16, shuffle=True, num_workers=0)
            
            self.dataset_cache[vehicle_id] = dataloader
            
            print(f"📦 {vehicle_id}: Created dataset with {len(subset)} samples")
        
        return self.dataset_cache[vehicle_id]
    
    def train_vehicle_model(self, vehicle_id: str, epochs: int = 1) -> dict:
        """Train model for a specific vehicle"""
        if vehicle_id not in self.vehicle_models:
            # Initialize vehicle model with global model weights
            self.vehicle_models[vehicle_id] = LightweightCNN(num_classes=10)
            if self.global_model:
                self.vehicle_models[vehicle_id].load_state_dict(
                    self.global_model.state_dict()
                )
        
        model = self.vehicle_models[vehicle_id]
        train_loader = self.get_vehicle_dataset(vehicle_id, num_samples=500)
        
        # Setup training
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        start_time = time.time()
        
        print(f"🚗 {vehicle_id}: Starting training ({epochs} epochs)...")
        
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
                
                # Memory cleanup
                if batch_idx % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            total_loss += epoch_loss
            correct += epoch_correct
            total_samples += epoch_total
            
            # Epoch statistics
            epoch_accuracy = 100. * epoch_correct / epoch_total if epoch_total > 0 else 0
            print(f"  {vehicle_id} Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, "
                  f"Acc={epoch_accuracy:.2f}%")
        
        training_time = time.time() - start_time
        final_accuracy = 100. * correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / (epochs * len(train_loader))
        
        # Calculate model update (parameter differences)
        model_update = self._calculate_model_update(model)
        
        # Store training statistics
        stats = {
            'vehicle_id': vehicle_id,
            'epochs': epochs,
            'training_time': training_time,
            'final_loss': avg_loss,
            'accuracy': final_accuracy,
            'samples_processed': total_samples,
            'model_update': model_update,
            'model_size': model.get_parameter_count()
        }
        
        self.training_stats[vehicle_id] = stats
        
        print(f"✅ {vehicle_id}: Training completed in {training_time:.2f}s, "
              f"Acc={final_accuracy:.2f}%")
        
        return stats
    
    def _calculate_model_update(self, local_model: nn.Module):
        """Calculate model update (differences from global model)"""
        if not self.global_model:
            # If no global model, return current parameters
            return {name: param.data.clone() for name, param in local_model.named_parameters()}
        
        # Calculate parameter differences
        update = {}
        local_params = dict(local_model.named_parameters())
        global_params = dict(self.global_model.named_parameters())
        
        for name in local_params:
            if name in global_params:
                update[name] = local_params[name].data - global_params[name].data
            else:
                update[name] = local_params[name].data.clone()
        
        return update
    
    def aggregate_updates(self, vehicle_stats: list):
        """Aggregate model updates from multiple vehicles"""
        if not vehicle_stats:
            return
        
        print(f"🔄 Aggregating updates from {len(vehicle_stats)} vehicles...")
        
        # Simple weighted averaging by number of samples
        aggregated_update = {}
        total_samples = sum(stat['samples_processed'] for stat in vehicle_stats)
        
        if total_samples == 0:
            return
        
        # Initialize aggregated update
        for name, param in vehicle_stats[0]['model_update'].items():
            aggregated_update[name] = torch.zeros_like(param)
        
        # Weighted sum
        for stat in vehicle_stats:
            weight = stat['samples_processed'] / total_samples
            for name, update in stat['model_update'].items():
                aggregated_update[name] += update * weight
        
        # Apply updates to global model
        if self.global_model:
            global_params = dict(self.global_model.named_parameters())
            for name, update in aggregated_update.items():
                if name in global_params:
                    global_params[name].data += update
            
            print("✅ Model aggregation completed")
        else:
            print("⚠️  No global model to aggregate to")

def create_test_vehicles(num_vehicles: int) -> list:
    """Create test vehicles with random configurations"""
    vehicles = []
    
    for i in range(num_vehicles):
        # Random position along highway
        x = random.uniform(-500, 500)
        y = random.uniform(-50, 50)
        
        # Random velocity (10-30 m/s = 36-108 km/h)
        velocity = random.uniform(10, 30)
        
        # Random direction (mostly forward with some variation)
        direction = random.uniform(-0.2, 0.2)
        
        # Random resources with more realistic distribution
        cpu = random.uniform(0.3, 0.9)
        memory = random.uniform(0.2, 0.8)
        battery = random.uniform(0.4, 1.0)
        
        vehicle_info = VehicleInfo(
            vehicle_id=f"vehicle_{i:03d}",
            position=(x, y),
            velocity=velocity,
            direction=direction,
            resources={
                'cpu': cpu,
                'memory': memory,
                'battery': battery,
                'network_quality': random.uniform(0.6, 1.0),
                'thermal_state': random.uniform(0.1, 0.5)
            },
            training_capability=cpu  # Use CPU as training capability score
        )
        
        vehicles.append(vehicle_info)
    
    return vehicles

def simulate_vehicle_movement(vehicle: Vehicle, duration: float, training_engine: RealTrainingEngine):
    """Simulate vehicle movement with real training"""
    start_time = time.time()
    training_count = 0
    
    while time.time() - start_time < duration:
        # Update position based on velocity and direction
        current_pos = vehicle.vehicle_info.position
        current_vel = vehicle.vehicle_info.velocity
        current_dir = vehicle.vehicle_info.direction
        
        # Simple linear movement
        dt = 0.1  # 100ms timestep
        new_x = current_pos[0] + current_vel * dt * 0.1
        new_y = current_pos[1] + current_vel * dt * 0.05
        
        # Wrap around boundaries
        if new_x > 1000:
            new_x = -1000
        if new_y > 100:
            new_y = -100
        
        vehicle.update_position((new_x, new_y), current_vel, current_dir)
        
        # Simulate resource changes
        new_cpu = max(0.2, min(0.9, vehicle.vehicle_info.resources['cpu'] + random.uniform(-0.05, 0.05)))
        new_memory = max(0.2, min(0.9, vehicle.vehicle_info.resources['memory'] + random.uniform(-0.03, 0.03)))
        new_battery = max(0.1, vehicle.vehicle_info.resources['battery'] - 0.001)
        
        vehicle.update_resources({
            'cpu': new_cpu,
            'memory': new_memory,
            'battery': new_battery
        })
        
        # Trigger training every 15 seconds (simulating training rounds)
        if int(time.time() - start_time) % 15 == 0 and training_count < 3:
            # Check if vehicle has enough resources for training
            if (new_cpu > 0.4 and new_memory > 0.3 and new_battery > 0.3):
                try:
                    epochs = 2 if new_battery > 0.7 else 1
                    training_engine.train_vehicle_model(vehicle.vehicle_info.vehicle_id, epochs=epochs)
                    training_count += 1
                except Exception as e:
                    print(f"❌ {vehicle.vehicle_info.vehicle_id}: Training failed - {e}")
        
        time.sleep(0.1)

def main():
    """Run enhanced FHDP simulation with real training"""
    print("=== FHDP Enhanced Simulation with Real Training ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Simulation parameters
    NUM_VEHICLES = 6  # Reduced for faster demonstration
    SIMULATION_DURATION = 60  # seconds
    
    print(f"🚗 Creating {NUM_VEHICLES} vehicles for {SIMULATION_DURATION}s simulation...")
    
    # Create FHDP system
    config = SystemConfiguration(
        max_vehicles_per_region=NUM_VEHICLES,
        pipeline_formation_interval=5.0,
        model_broadcast_interval=10.0,
        enable_pipeline_training=True,
        enable_individual_training=True,
        fairness_enabled=True
    )
    
    system = FHDPSystem(config)
    system.start_system()
    
    # Create edge server
    edge_server = EdgeServer()
    edge_server.start_server()
    
    # Create real training engine
    training_engine = RealTrainingEngine()
    training_engine.prepare_global_model()
    
    # Create vehicles
    vehicle_infos = create_test_vehicles(NUM_VEHICLES)
    vehicles = []
    
    print("\\n📊 Vehicle Configuration:")
    for i, v_info in enumerate(vehicle_infos):
        gpu_available = v_info.resources['cpu'] > 0.6
        print(f"  {v_info.vehicle_id}: pos=({v_info.position[0]:.1f},{v_info.position[1]:.1f}), "
              f"vel={v_info.velocity:.1f}m/s, "
              f"cpu={v_info.resources['cpu']:.2f}, "
              f"mem={v_info.resources['memory']:.2f}, "
              f"bat={v_info.resources['battery']:.2f}, "
              f"gpu={gpu_available}")
        
        # Create vehicle
        vehicle = Vehicle(
            vehicle_id=v_info.vehicle_id,
            initial_position=v_info.position,
            initial_velocity=v_info.velocity,
            initial_direction=v_info.direction,
            resources=v_info.resources
        )
        
        # Start vehicle
        vehicle.start_vehicle(['dsrc'])
        
        # Register with system and edge server
        system.register_vehicle(v_info)
        edge_server.register_vehicle(v_info)
        
        vehicles.append(vehicle)
    
    print("\\n🧠 Starting enhanced simulation with real neural network training...")
    
    # Start movement threads for all vehicles
    movement_threads = []
    for vehicle in vehicles:
        thread = threading.Thread(
            target=simulate_vehicle_movement,
            args=(vehicle, SIMULATION_DURATION, training_engine)
        )
        thread.daemon = True
        thread.start()
        movement_threads.append(thread)
    
    # Main simulation loop with periodic aggregation
    start_time = time.time()
    last_aggregation_time = start_time
    last_status_time = start_time
    
    try:
        while time.time() - start_time < SIMULATION_DURATION:
            current_time = time.time()
            
            # Periodic model aggregation every 20 seconds
            if current_time - last_aggregation_time >= 20:
                if training_engine.training_stats:
                    print("\\n🔄 === Model Aggregation ===")
                    vehicle_stats = list(training_engine.training_stats.values())
                    training_engine.aggregate_updates(vehicle_stats)
                    
                    # Clear stats for next round
                    training_engine.training_stats.clear()
                    last_aggregation_time = current_time
            
            # Print status every 10 seconds
            if current_time - last_status_time >= 10:
                system_status = system.get_system_status()
                server_status = edge_server.get_server_statistics()
                
                print(f"\\n--- Status at t={current_time - start_time:.1f}s ---")
                print(f"Active vehicles: {system_status['registered_vehicles']}")
                print(f"Active pipelines: {system_status['active_pipelines']}")
                print(f"Training rounds: {system_status['round_number']}")
                print(f"Total aggregations: {server_status['aggregations_performed']}")
                
                # Show training statistics
                if training_engine.training_stats:
                    avg_accuracy = np.mean([s['accuracy'] for s in training_engine.training_stats.values()])
                    total_samples = sum([s['samples_processed'] for s in training_engine.training_stats.values()])
                    print(f"Recent training: {len(training_engine.training_stats)} vehicles, "
                          f"Avg accuracy: {avg_accuracy:.2f}%, "
                          f"Total samples: {total_samples}")
                
                last_status_time = current_time
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\nSimulation interrupted by user")
    
    finally:
        print("\\n🛑 Shutting down simulation...")
        
        # Stop all vehicles
        for vehicle in vehicles:
            vehicle.stop_vehicle()
        
        # Stop system and server
        system.stop_system()
        edge_server.stop_server()
        
        # Wait for movement threads to finish
        for thread in movement_threads:
            thread.join(timeout=2.0)
        
        # Final aggregation
        if training_engine.training_stats:
            print("\\n🔄 Final Model Aggregation")
            vehicle_stats = list(training_engine.training_stats.values())
            training_engine.aggregate_updates(vehicle_stats)
        
        # Print final statistics
        print("\\n=== Final Statistics ===")
        final_status = system.get_system_status()
        final_server_stats = edge_server.get_server_statistics()
        
        print(f"Total training rounds: {final_status['round_number']}")
        print(f"Total vehicles served: {final_status['total_vehicles_served']}")
        print(f"Total pipelines formed: {final_status['total_pipelines_formed']}")
        print(f"Total aggregations: {final_server_stats['aggregations_performed']}")
        print(f"System uptime: {final_status['uptime']:.1f}s")
        
        # Enhanced training statistics
        all_training_stats = training_engine.training_stats
        if all_training_stats:
            print("\\n🧠 Neural Network Training Results:")
            
            total_samples = sum([s['samples_processed'] for s in all_training_stats.values()])
            total_training_time = sum([s['training_time'] for s in all_training_stats.values()])
            avg_accuracy = np.mean([s['accuracy'] for s in all_training_stats.values()])
            avg_loss = np.mean([s['final_loss'] for s in all_training_stats.values()])
            
            print(f"  Total samples processed: {total_samples:,}")
            print(f"  Total training time: {total_training_time:.2f}s")
            print(f"  Average accuracy: {avg_accuracy:.2f}%")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Number of training sessions: {len(all_training_stats)}")
            
            print("\\n📈 Individual Vehicle Performance:")
            for vehicle_id, stats in all_training_stats.items():
                print(f"  {vehicle_id}:")
                print(f"    Sessions: {stats['epochs']} epochs")
                print(f"    Accuracy: {stats['accuracy']:.2f}%")
                print(f"    Training time: {stats['training_time']:.2f}s")
                print(f"    Samples: {stats['samples_processed']}")
                print(f"    Model size: {stats['model_size']:,} parameters")
        
        print("\\n✅ Enhanced simulation with real training completed successfully!")

if __name__ == '__main__':
    main()