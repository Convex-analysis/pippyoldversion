#!/usr/bin/env python3
"""
Simple FHDP Simulation Example

Demonstrates basic FHDP system functionality with a small simulation.
"""
import sys
import os
import time
import random
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import FHDPSystem, SystemConfiguration
from edge_server import EdgeServer
from vehicle_layer import Vehicle
from core.types import VehicleInfo, ResourceClass

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
        
        # Random resources
        cpu = random.uniform(0.4, 0.9)
        memory = random.uniform(0.3, 0.8)
        battery = random.uniform(0.3, 1.0)
        
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
            }
        )
        
        vehicles.append(vehicle_info)
    
    return vehicles

def simulate_vehicle_movement(vehicle: Vehicle, duration: float):
    """Simulate vehicle movement during simulation"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Update position based on velocity and direction
        current_pos = vehicle.vehicle_info.position
        current_vel = vehicle.vehicle_info.velocity
        current_dir = vehicle.vehicle_info.direction
        
        # Simple linear movement
        dt = 0.1  # 100ms timestep
        new_x = current_pos[0] + current_vel * dt * 0.1  # Scale down for simulation
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
        new_battery = max(0.1, vehicle.vehicle_info.resources['battery'] - 0.001)  # Slow battery drain
        
        vehicle.update_resources({
            'cpu': new_cpu,
            'memory': new_memory,
            'battery': new_battery
        })
        
        time.sleep(0.1)

def main():
    """Run simple FHDP simulation"""
    print("=== FHDP Simple Simulation ===")
    
    # Simulation parameters
    NUM_VEHICLES = 8
    SIMULATION_DURATION = 60  # seconds
    
    print(f"Creating {NUM_VEHICLES} vehicles for {SIMULATION_DURATION}s simulation...")
    
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
    
    # Create vehicles
    vehicle_infos = create_test_vehicles(NUM_VEHICLES)
    vehicles = []
    
    print("\\nVehicle Configuration:")
    for i, v_info in enumerate(vehicle_infos):
        print(f"  {v_info.vehicle_id}: pos=({v_info.position[0]:.1f},{v_info.position[1]:.1f}), "
              f"vel={v_info.velocity:.1f}m/s, "
              f"resources=cpu:{v_info.resources['cpu']:.2f}, "
              f"mem:{v_info.resources['memory']:.2f}, "
              f"bat:{v_info.resources['battery']:.2f}")
        
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
    
    print("\\nStarting simulation...")
    
    # Start movement threads for all vehicles
    movement_threads = []
    for vehicle in vehicles:
        thread = threading.Thread(
            target=simulate_vehicle_movement,
            args=(vehicle, SIMULATION_DURATION)
        )
        thread.daemon = True
        thread.start()
        movement_threads.append(thread)
    
    # Main simulation loop
    start_time = time.time()
    last_status_time = start_time
    
    try:
        while time.time() - start_time < SIMULATION_DURATION:
            current_time = time.time()
            
            # Print status every 10 seconds
            if current_time - last_status_time >= 10:
                system_status = system.get_system_status()
                server_status = edge_server.get_server_statistics()
                
                print(f"\\n--- Status at t={current_time - start_time:.1f}s ---")
                print(f"Active vehicles: {system_status['registered_vehicles']}")
                print(f"Active pipelines: {system_status['active_pipelines']}")
                print(f"Training rounds: {system_status['round_number']}")
                print(f"Total aggregations: {server_status['aggregations_performed']}")
                
                # Show some vehicle details
                print("\\nSample vehicle states:")
                for i, vehicle in enumerate(vehicles[:3]):  # Show first 3
                    status = vehicle.get_vehicle_status()
                    print(f"  {vehicle.vehicle_info.vehicle_id}: {status['current_state']}, "
                          f"{status['neighbors_count']} neighbors, "
                          f"{status['total_training_sessions']} sessions")
                
                last_status_time = current_time
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\nSimulation interrupted by user")
    
    finally:
        print("\\nShutting down simulation...")
        
        # Stop all vehicles
        for vehicle in vehicles:
            vehicle.stop_vehicle()
        
        # Stop system and server
        system.stop_system()
        edge_server.stop_server()
        
        # Wait for movement threads to finish
        for thread in movement_threads:
            thread.join(timeout=2.0)
        
        # Print final statistics
        print("\\n=== Final Statistics ===")
        final_status = system.get_system_status()
        final_server_stats = edge_server.get_server_statistics()
        
        print(f"Total training rounds: {final_status['round_number']}")
        print(f"Total vehicles served: {final_status['total_vehicles_served']}")
        print(f"Total pipelines formed: {final_status['total_pipelines_formed']}")
        print(f"Total aggregations: {final_status['total_aggregations']}")
        print(f"System uptime: {final_status['uptime']:.1f}s")
        
        # Vehicle statistics
        total_sessions = sum(v.get_vehicle_status()['total_training_sessions'] for v in vehicles)
        avg_sessions = total_sessions / len(vehicles)
        print(f"Average training sessions per vehicle: {avg_sessions:.1f}")
        
        print("\\nSimulation completed successfully!")

if __name__ == '__main__':
    main()