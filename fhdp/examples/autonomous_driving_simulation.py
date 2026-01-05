#!/usr/bin/env python3
"""
Autonomous Driving Simulation with EVO-1 Model and FHDP Integration

This simulation integrates the EVO-1 Vision-Language-Action model with FHDP
for federated learning in autonomous driving scenarios using nuScenes dataset.
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
import asyncio
import websockets
import json
import cv2
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fhdp.core import FHDPSystem, SystemConfiguration
from fhdp.edge_server import EdgeServer
from fhdp.vehicle_layer import Vehicle
from fhdp.core.types import VehicleInfo, TrainingConfig, TrainingMode

# EVO-1 Model Components
class EVO1Observation:
    """EVO-1 compatible observation format"""
    def __init__(self):
        self.images = []  # List of camera images (6 cameras for nuScenes)
        self.image_masks = []  # Valid camera indicators
        self.state = []  # Vehicle state [speed, yaw, acceleration, etc.]
        self.action_mask = []  # Action dimension validity
        self.prompt = ""  # Language instruction

class EVO1ModelClient:
    """Client for EVO-1 model server communication"""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to EVO-1 server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            print(f"🔗 Connected to EVO-1 server at {self.server_url}")
        except Exception as e:
            print(f"⚠️  Failed to connect to EVO-1 server: {e}")
            self.is_connected = False
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
    
    async def predict_action(self, observation: EVO1Observation) -> np.ndarray:
        """Send observation and get action prediction"""
        if not self.is_connected:
            # Fallback to simple policy if server not available
            return self._fallback_policy(observation)
        
        try:
            # Prepare observation for EVO-1
            obs_dict = {
                "image": self._prepare_images(observation.images),
                "image_mask": observation.image_masks,
                "state": observation.state,
                "action_mask": observation.action_mask,
                "prompt": observation.prompt
            }
            
            await self.websocket.send(json.dumps(obs_dict))
            response = await self.websocket.recv()
            action_data = json.loads(response)
            
            return np.array(action_data["action"])
        except Exception as e:
            print(f"⚠️  EVO-1 prediction failed, using fallback: {e}")
            return self._fallback_policy(observation)
    
    def _prepare_images(self, images: List[np.ndarray]) -> List[List]:
        """Prepare images for EVO-1 (resize to 448x448)"""
        prepared = []
        for img in images:
            if img is not None:
                # Resize to 448x448 as required by EVO-1
                if img.shape[:2] != (448, 448):
                    img = cv2.resize(img, (448, 448))
                # Normalize and convert to list format
                img_normalized = (img.astype(np.float32) / 255.0).tolist()
                prepared.append(img_normalized)
            else:
                prepared.append([])
        return prepared
    
    def _fallback_policy(self, observation: EVO1Observation) -> np.ndarray:
        """Simple fallback policy for autonomous driving"""
        # Simple lane keeping policy based on current state
        if len(observation.state) >= 3:
            speed, yaw, accel = observation.state[:3]
            
            # Basic control: maintain speed, slight steering correction
            steering = -yaw * 0.1  # Simple proportional control
            throttle = 0.5 if speed < 20 else 0.0  # Maintain ~20 m/s
            brake = 0.0
            
            return np.array([steering, throttle, brake])
        
        return np.array([0.0, 0.5, 0.0])  # Default: straight, half throttle

class NuScenesDataLoader:
    """Simplified nuScenes data processor for simulation"""
    
    def __init__(self, data_root: str = "./nuscenes_data"):
        self.data_root = data_root
        self.current_scene_idx = 0
        self.current_frame_idx = 0
        self.scenes = []
        self.is_loaded = False
        
        # Simulated data for demonstration
        self.simulated_scenes = self._create_simulated_scenes()
    
    def _create_simulated_scenes(self) -> List[Dict]:
        """Create simulated driving scenarios for testing"""
        scenarios = []
        
        # Highway driving scenario
        highway = {
            "name": "highway_driving",
            "description": "Highway cruising with lane changes",
            "duration": 300,  # 5 minutes
            "weather": "clear",
            "time_of_day": "day",
            "traffic_density": "medium"
        }
        scenarios.append(highway)
        
        # Urban intersection scenario
        urban = {
            "name": "urban_intersection",
            "description": "Complex urban navigation with traffic lights",
            "duration": 180,  # 3 minutes
            "weather": "cloudy",
            "time_of_day": "day",
            "traffic_density": "high"
        }
        scenarios.append(urban)
        
        # Night driving scenario
        night = {
            "name": "night_driving",
            "description": "Night highway driving with reduced visibility",
            "duration": 240,  # 4 minutes
            "weather": "clear",
            "time_of_day": "night",
            "traffic_density": "low"
        }
        scenarios.append(night)
        
        return scenarios
    
    def load_scene(self, scene_idx: int = 0):
        """Load a specific driving scenario"""
        if scene_idx < len(self.simulated_scenes):
            self.current_scene = self.simulated_scenes[scene_idx]
            self.current_frame_idx = 0
            self.is_loaded = True
            print(f"📍 Loaded scene: {self.current_scene['name']}")
        else:
            print(f"❌ Scene {scene_idx} not found")
    
    def get_next_frame(self) -> Optional[EVO1Observation]:
        """Get next frame from current scene"""
        if not self.is_loaded:
            return None
        
        # Simulate camera data (6 cameras around vehicle)
        images = []
        image_masks = []
        
        for cam_idx in range(6):
            # Generate simulated camera image
            img = self._generate_simulated_camera_image(cam_idx)
            images.append(img)
            image_masks.append(1)  # All cameras valid
        
        # Simulate vehicle state
        state = self._generate_simulated_state()
        
        # Generate appropriate prompt based on scenario
        prompt = self._generate_driving_prompt()
        
        obs = EVO1Observation()
        obs.images = images
        obs.image_masks = image_masks
        obs.state = state
        obs.action_mask = [[1, 1, 1]]  # All actions valid
        obs.prompt = prompt
        
        self.current_frame_idx += 1
        return obs
    
    def _generate_simulated_camera_image(self, cam_idx: int) -> np.ndarray:
        """Generate simulated camera view"""
        # Create a simple synthetic image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure based on camera position
        if cam_idx == 0:  # Front camera
            # Simulate road and horizon
            img[240:, :] = [80, 80, 80]  # Road
            img[:240, :] = [135, 206, 235]  # Sky
        
        return img
    
    def _generate_simulated_state(self) -> List[float]:
        """Generate simulated vehicle state"""
        # [speed (m/s), yaw (rad), acceleration (m/s²), steering_angle (rad)]
        speed = random.uniform(10, 30)  # 10-30 m/s
        yaw = random.uniform(-0.1, 0.1)  # Small yaw variation
        accel = random.uniform(-2, 2)  # Acceleration
        steering = random.uniform(-0.3, 0.3)  # Steering angle
        
        return [speed, yaw, accel, steering]
    
    def _generate_driving_prompt(self) -> str:
        """Generate appropriate driving instruction"""
        prompts = [
            "保持车道并平稳行驶",
            "准备变道到左侧车道",
            "注意前方车辆并保持安全距离",
            "在交叉路口准备左转",
            "夜间模式：提高警觉，降低车速"
        ]
        return random.choice(prompts)

class AutonomousDrivingVehicle:
    """Enhanced vehicle with EVO-1 autonomous driving capabilities"""
    
    def __init__(self, vehicle_id: str, initial_position: Tuple[float, float], 
                 evo1_client: EVO1ModelClient, data_loader: NuScenesDataLoader):
        self.vehicle_id = vehicle_id
        self.position = initial_position
        self.evo1_client = evo1_client
        self.data_loader = data_loader
        
        # Vehicle state
        self.velocity = 0.0
        self.direction = 0.0
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        
        # Driving metrics
        self.total_distance = 0.0
        self.safety_violations = 0
        self.comfort_score = 100.0
        self.efficiency_score = 100.0
        
        # Thread management
        self.driving_thread = None
        self.is_driving = False
    
    async def start_autonomous_driving(self, duration: float):
        """Start autonomous driving loop"""
        self.is_driving = True
        self.driving_thread = asyncio.create_task(self._driving_loop(duration))
    
    async def _driving_loop(self, duration: float):
        """Main driving loop"""
        start_time = time.time()
        
        while time.time() - start_time < duration and self.is_driving:
            try:
                # Get current observation
                obs = self.data_loader.get_next_frame()
                if obs is None:
                    print(f"⚠️  {self.vehicle_id}: No more data available")
                    break
                
                # Get action from EVO-1
                action = await self.evo1_client.predict_action(obs)
                
                # Apply action to vehicle
                self._apply_action(action)
                
                # Update metrics
                self._update_metrics(action, obs)
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"❌ {self.vehicle_id}: Driving error - {e}")
                break
    
    def _apply_action(self, action: np.ndarray):
        """Apply EVO-1 action to vehicle"""
        if len(action) >= 3:
            self.steering = np.clip(action[0], -1.0, 1.0)
            self.throttle = np.clip(action[1], 0.0, 1.0)
            self.brake = np.clip(action[2], 0.0, 1.0)
            
            # Update vehicle physics (simplified)
            self.velocity += (self.throttle - self.brake) * 2.0
            self.velocity = max(0, min(40, self.velocity))  # 0-40 m/s
            
            self.direction += self.steering * 0.05
            
            # Update position
            self.position = (
                self.position[0] + self.velocity * np.cos(self.direction) * 0.1,
                self.position[1] + self.velocity * np.sin(self.direction) * 0.1
            )
            
            self.total_distance += self.velocity * 0.1
    
    def _update_metrics(self, action: np.ndarray, obs: EVO1Observation):
        """Update driving performance metrics"""
        # Safety: check for aggressive maneuvers
        if abs(self.steering) > 0.5 or self.brake > 0.8:
            self.safety_violations += 1
        
        # Comfort: penalize jerky movements
        jerk = abs(self.steering) + abs(self.throttle - self.brake)
        self.comfort_score = max(0, self.comfort_score - jerk * 0.1)
        
        # Efficiency: reward steady driving
        if 15 <= self.velocity <= 25:  # Optimal speed range
            self.efficiency_score = min(100, self.efficiency_score + 0.05)
    
    def stop_driving(self):
        """Stop autonomous driving"""
        self.is_driving = False
        if self.driving_thread:
            self.driving_thread.cancel()
    
    def get_driving_report(self) -> Dict:
        """Get comprehensive driving performance report"""
        return {
            "vehicle_id": self.vehicle_id,
            "total_distance": self.total_distance,
            "average_speed": self.total_distance / max(1, self.total_distance / self.velocity),
            "safety_violations": self.safety_violations,
            "comfort_score": self.comfort_score,
            "efficiency_score": self.efficiency_score,
            "final_position": self.position,
            "final_velocity": self.velocity
        }

class FederatedEVO1Trainer:
    """Federated learning trainer for EVO-1 model with FHDP"""
    
    def __init__(self):
        self.vehicle_models = {}
        self.global_model_weights = None
        self.training_stats = {}
        
    def prepare_vehicle_model(self, vehicle_id: str):
        """Prepare EVO-1 model for a vehicle"""
        # In real implementation, this would load EVO-1 model
        # For simulation, we create a mock model
        model = {
            "integration_module": torch.randn(8, 256, 256),  # Mock weights
            "action_head": torch.randn(256, 3),  # Steering, throttle, brake
            "training_round": 0
        }
        
        if self.global_model_weights:
            # Load global model weights
            model["integration_module"] = self.global_model_weights["integration_module"].clone()
            model["action_head"] = self.global_model_weights["action_head"].clone()
        
        self.vehicle_models[vehicle_id] = model
        print(f"🧠 {vehicle_id}: EVO-1 model prepared")
    
    def train_vehicle_model(self, vehicle_id: str, observations: List, actions: List) -> Dict:
        """Train EVO-1 model on vehicle's driving data"""
        if vehicle_id not in self.vehicle_models:
            self.prepare_vehicle_model(vehicle_id)
        
        model = self.vehicle_models[vehicle_id]
        
        # Mock training process
        start_time = time.time()
        
        # Simulate training epochs
        for epoch in range(3):  # 3 epochs for demo
            loss = random.uniform(0.1, 0.5)  # Mock loss
            
            # Simulate weight updates
            noise = torch.randn_like(model["integration_module"]) * 0.01
            model["integration_module"] += noise
            model["action_head"] += torch.randn_like(model["action_head"]) * 0.01
        
        training_time = time.time() - start_time
        model["training_round"] += 1
        
        stats = {
            "vehicle_id": vehicle_id,
            "training_time": training_time,
            "epochs": 3,
            "samples_trained": len(observations),
            "final_loss": loss,
            "model_update": {
                "integration_module": noise,
                "action_head": torch.randn_like(model["action_head"]) * 0.01
            }
        }
        
        self.training_stats[vehicle_id] = stats
        print(f"✅ {vehicle_id}: EVO-1 training completed in {training_time:.2f}s")
        
        return stats
    
    def aggregate_model_updates(self, vehicle_ids: List[str]):
        """Aggregate model updates from multiple vehicles using federated learning"""
        if not vehicle_ids:
            return
        
        print(f"🔄 Aggregating EVO-1 updates from {len(vehicle_ids)} vehicles...")
        
        # Federated averaging
        integration_updates = []
        action_updates = []
        
        for vehicle_id in vehicle_ids:
            if vehicle_id in self.training_stats:
                update = self.training_stats[vehicle_id]["model_update"]
                integration_updates.append(update["integration_module"])
                action_updates.append(update["action_head"])
        
        if integration_updates and action_updates:
            # Average the updates
            avg_integration = torch.stack(integration_updates).mean(dim=0)
            avg_action = torch.stack(action_updates).mean(dim=0)
            
            # Update global model
            self.global_model_weights = {
                "integration_module": avg_integration,
                "action_head": avg_action
            }
            
            print("✅ EVO-1 model aggregation completed")

def create_autonomous_vehicles(num_vehicles: int) -> List[AutonomousDrivingVehicle]:
    """Create autonomous vehicles with EVO-1 integration"""
    vehicles = []
    
    # Shared components
    evo1_client = EVO1ModelClient()
    data_loader = NuScenesDataLoader()
    
    for i in range(num_vehicles):
        vehicle_id = f"evo1_vehicle_{i:03d}"
        
        # Random starting position
        x = random.uniform(-100, 100)
        y = random.uniform(-50, 50)
        initial_position = (x, y)
        
        # Create autonomous vehicle
        vehicle = AutonomousDrivingVehicle(
            vehicle_id=vehicle_id,
            initial_position=initial_position,
            evo1_client=evo1_client,
            data_loader=data_loader
        )
        
        vehicles.append(vehicle)
    
    return vehicles

async def main():
    """Run autonomous driving simulation with EVO-1 and FHDP"""
    print("=== FHDP Autonomous Driving Simulation with EVO-1 ===")
    
    # Simulation parameters
    NUM_VEHICLES = 4
    SIMULATION_DURATION = 120  # seconds
    
    print(f"🚗 Creating {NUM_VEHICLES} autonomous vehicles for {SIMULATION_DURATION}s simulation...")
    
    # Create FHDP system
    config = SystemConfiguration(
        max_vehicles_per_region=NUM_VEHICLES,
        pipeline_formation_interval=10.0,
        model_broadcast_interval=15.0,
        enable_pipeline_training=True,
        enable_individual_training=True,
        fairness_enabled=True
    )
    
    system = FHDPSystem(config)
    system.start_system()
    
    # Create edge server
    edge_server = EdgeServer()
    edge_server.start_server()
    
    # Create EVO-1 trainer
    evo1_trainer = FederatedEVO1Trainer()
    
    # Create autonomous vehicles
    autonomous_vehicles = create_autonomous_vehicles(NUM_VEHICLES)
    
    # Initialize data loader
    data_loader = NuScenesDataLoader()
    data_loader.load_scene(0)  # Load first scene
    
    # Connect to EVO-1 server
    evo1_client = EVO1ModelClient()
    await evo1_client.connect()
    
    print("\n🧠 Starting autonomous driving with EVO-1...")
    
    # Start driving tasks
    driving_tasks = []
    for vehicle in autonomous_vehicles:
        task = asyncio.create_task(
            vehicle.start_autonomous_driving(SIMULATION_DURATION)
        )
        driving_tasks.append(task)
    
    # Main simulation loop
    start_time = time.time()
    last_training_time = start_time
    last_aggregation_time = start_time
    
    try:
        while time.time() - start_time < SIMULATION_DURATION:
            current_time = time.time()
            
            # Periodic training every 30 seconds
            if current_time - last_training_time >= 30:
                print("\n🎓 === Federated Training Round ===")
                
                for vehicle in autonomous_vehicles:
                    # Collect observations and actions (simplified)
                    observations = [data_loader.get_next_frame() for _ in range(10)]
                    actions = [np.random.randn(3) for _ in range(10)]  # Mock actions
                    
                    evo1_trainer.train_vehicle_model(
                        vehicle.vehicle_id, 
                        observations, 
                        actions
                    )
                
                last_training_time = current_time
            
            # Periodic aggregation every 45 seconds
            if current_time - last_aggregation_time >= 45:
                print("\n🔄 === Model Aggregation ===")
                vehicle_ids = [v.vehicle_id for v in autonomous_vehicles]
                evo1_trainer.aggregate_model_updates(vehicle_ids)
                last_aggregation_time = current_time
            
            # Print status every 15 seconds
            if int(current_time - start_time) % 15 == 0:
                print(f"\n--- Status at t={current_time - start_time:.1f}s ---")
                
                system_status = system.get_system_status()
                server_status = edge_server.get_server_statistics()
                
                print(f"Active vehicles: {system_status['registered_vehicles']}")
                print(f"Training rounds: {system_status['round_number']}")
                print(f"Aggregations: {server_status['aggregations_performed']}")
                
                # Show driving statistics
                total_distance = sum(v.total_distance for v in autonomous_vehicles)
                avg_comfort = np.mean([v.comfort_score for v in autonomous_vehicles])
                total_violations = sum(v.safety_violations for v in autonomous_vehicles)
                
                print(f"Total distance: {total_distance:.1f}m")
                print(f"Avg comfort score: {avg_comfort:.1f}")
                print(f"Safety violations: {total_violations}")
            
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        print("\n🛑 Shutting down autonomous driving simulation...")
        
        # Stop all vehicles
        for vehicle in autonomous_vehicles:
            vehicle.stop_driving()
        
        # Cancel driving tasks
        for task in driving_tasks:
            task.cancel()
        
        # Stop system and server
        system.stop_system()
        edge_server.stop_server()
        
        # Disconnect from EVO-1 server
        await evo1_client.disconnect()
        
        # Print final statistics
        print("\n=== Final Autonomous Driving Statistics ===")
        
        for vehicle in autonomous_vehicles:
            report = vehicle.get_driving_report()
            print(f"\n📊 {report['vehicle_id']}:")
            print(f"  Total distance: {report['total_distance']:.1f}m")
            print(f"  Average speed: {report['average_speed']:.1f} m/s")
            print(f"  Safety violations: {report['safety_violations']}")
            print(f"  Comfort score: {report['comfort_score']:.1f}")
            print(f"  Efficiency score: {report['efficiency_score']:.1f}")
        
        # EVO-1 training statistics
        if evo1_trainer.training_stats:
            print("\n🧠 EVO-1 Federated Learning Results:")
            for vehicle_id, stats in evo1_trainer.training_stats.items():
                print(f"  {vehicle_id}:")
                print(f"    Training time: {stats['training_time']:.2f}s")
                print(f"    Samples trained: {stats['samples_trained']}")
                print(f"    Final loss: {stats['final_loss']:.4f}")
        
        print("\n✅ Autonomous driving simulation with EVO-1 completed successfully!")

if __name__ == '__main__':
    asyncio.run(main())