#!/usr/bin/env python3
"""
FHDP System Main Entry Point

Main entry point for the FHDP (Federated Highway-based Distributed Pipeline) system.
Provides command-line interface for running edge servers, vehicles, and simulations.
"""
import argparse
import sys
import time
import signal
import threading
from pathlib import Path

from .core import FHDPSystem, SystemConfiguration
from .edge_server import EdgeServer
from .vehicle_layer import Vehicle
from .core.types import CommunicationProtocol

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutting down FHDP system...")
    sys.exit(0)

def run_edge_server(args):
    """Run edge server"""
    print(f"Starting FHDP Edge Server on port {args.port}")
    
    # Create edge server
    server = EdgeServer(args.config if args.config else None)
    
    # Set coverage area
    if args.coverage:
        width, height = map(float, args.coverage.split('x'))
        server.set_coverage_area(width, height)
    
    # Start server
    server.start_server()
    
    try:
        # Keep server running
        while True:
            time.sleep(1)
            
            # Print status periodically
            if int(time.time()) % 30 == 0:
                stats = server.get_server_statistics()
                print(f"Server Status: {stats['registered_vehicles']} vehicles, "
                      f"{stats['aggregations_performed']} aggregations")
                
    except KeyboardInterrupt:
        server.stop_server()
        print("Edge server stopped")

def run_vehicle(args):
    """Run vehicle"""
    print(f"Starting FHDP Vehicle {args.vehicle_id}")
    
    # Parse initial position
    if args.position:
        x, y = map(float, args.position.split(','))
        position = (x, y)
    else:
        position = (0, 0)
    
    # Parse protocols
    protocols = args.protocols.split(',') if args.protocols else ['dsrc']
    
    # Create vehicle
    vehicle = Vehicle(
        vehicle_id=args.vehicle_id,
        initial_position=position,
        initial_velocity=args.velocity,
        initial_direction=args.direction
    )
    
    # Start vehicle
    vehicle.start_vehicle(protocols)
    
    try:
        # Keep vehicle running
        while True:
            time.sleep(1)
            
            # Update position (simulate movement)
            if args.simulate_movement:
                current_time = time.time()
                x = position[0] + args.velocity * (current_time % 10) * 0.1
                y = position[1] + args.velocity * (current_time % 10) * 0.05
                vehicle.update_position((x, y), args.velocity, 0)
            
            # Print status periodically
            if int(time.time()) % 15 == 0:
                status = vehicle.get_vehicle_status()
                print(f"Vehicle {args.vehicle_id} Status: {status['current_state']}, "
                      f"{status['neighbors_count']} neighbors, "
                      f"{status['total_training_sessions']} training sessions")
                
    except KeyboardInterrupt:
        vehicle.stop_vehicle()
        print(f"Vehicle {args.vehicle_id} stopped")

def run_simulation(args):
    """Run FHDP simulation"""
    print(f"Starting FHDP Simulation with {args.num_vehicles} vehicles")
    
    # Create system configuration
    config = SystemConfiguration(
        max_vehicles_per_region=args.num_vehicles,
        pipeline_formation_interval=5.0,
        enable_pipeline_training=True,
        enable_individual_training=True,
        fairness_enabled=True
    )
    
    # Create FHDP system
    system = FHDPSystem(config)
    
    # Start system
    system.start_system()
    
    # Create edge server
    edge_server = EdgeServer()
    
    # Set coverage area (1000x1000 meters centered at origin)
    # This allows vehicles with positions from -500 to 500 in both x and y
    edge_server.set_coverage_area(1000, 1000)
    edge_server.start_server()
    
    # Create vehicles
    vehicles = []
    for i in range(args.num_vehicles):
        # Random initial positions
        import random
        x = random.uniform(-500, 500)
        y = random.uniform(-500, 500)
        velocity = random.uniform(10, 30)  # 10-30 m/s
        
        vehicle = Vehicle(
            vehicle_id=f"vehicle_{i:03d}",
            initial_position=(x, y),
            initial_velocity=velocity,
            initial_direction=random.uniform(0, 2 * 3.14159)
        )
        
        # Start vehicle
        vehicle.start_vehicle(['dsrc'])
        vehicles.append(vehicle)
        
        # Register with edge server
        vehicle_info = vehicle.vehicle_info
        edge_server.register_vehicle(vehicle_info)
        system.register_vehicle(vehicle_info)
        
        print(f"Created vehicle {i:03d} at ({x:.1f}, {y:.1f})")
    
    try:
        # Run simulation
        simulation_time = 0
        while simulation_time < args.duration:
            time.sleep(1)
            simulation_time += 1
            
            # Update vehicle positions (simulate highway movement)
            for i, vehicle in enumerate(vehicles):
                # Simple linear movement
                current_pos = vehicle.vehicle_info.position
                new_x = current_pos[0] + vehicle.vehicle_info.velocity * 0.1  # Move forward
                new_y = current_pos[1] + (i - args.num_vehicles/2) * 0.1  # Slight lateral drift
                
                # Wrap around
                if new_x > 1000:
                    new_x = -1000
                if new_y > 500 or new_y < -500:
                    new_y = 0
                
                vehicle.update_position((new_x, new_y), vehicle.vehicle_info.velocity, 0)
                
                # Update edge server
                edge_server.update_vehicle_position(
                    vehicle.vehicle_info.vehicle_id,
                    (new_x, new_y),
                    vehicle.vehicle_info.velocity,
                    0
                )
            
            # Print status every 10 seconds
            if simulation_time % 10 == 0:
                system_status = system.get_system_status()
                server_status = edge_server.get_server_statistics()
                
                print(f"Simulation Time: {simulation_time}s | "
                      f"Active Vehicles: {system_status['registered_vehicles']} | "
                      f"Active Pipelines: {system_status['active_pipelines']} | "
                      f"Total Aggregations: {server_status['aggregations_performed']}")
                
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    
    finally:
        # Cleanup
        print("Shutting down simulation...")
        
        for vehicle in vehicles:
            vehicle.stop_vehicle()
        
        system.stop_system()
        edge_server.stop_server()
        
        # Print final statistics
        final_status = system.get_system_status()
        print("\n=== Final Simulation Statistics ===")
        print(f"Total Rounds: {final_status['round_number']}")
        print(f"Total Vehicles Served: {final_status['total_vehicles_served']}")
        print(f"Total Pipelines Formed: {final_status['total_pipelines_formed']}")
        print(f"Total Aggregations: {final_status['total_aggregations']}")
        print(f"System Uptime: {final_status['uptime']:.1f} seconds")

def run_benchmark(args):
    """Run FHDP performance benchmark"""
    print("Running FHDP Performance Benchmark")
    
    # Benchmark different aspects
    benchmarks = {
        'template_lookup': benchmark_template_lookup,
        'pipeline_formation': benchmark_pipeline_formation,
        'aggregation': benchmark_aggregation,
        'communication': benchmark_communication
    }
    
    results = {}
    
    for benchmark_name, benchmark_func in benchmarks.items():
        print(f"\nRunning {benchmark_name} benchmark...")
        results[benchmark_name] = benchmark_func()
        print(f"{benchmark_name}: {results[benchmark_name]}")
    
    print("\n=== Benchmark Results ===")
    for name, result in results.items():
        print(f"{name}: {result}")

def benchmark_template_lookup():
    """Benchmark template lookup performance"""
    from .edge_server import TemplateManager
    from .core.types import VehicleInfo
    
    manager = TemplateManager()
    
    # Create test vehicles
    vehicles = []
    for i in range(100):
        vehicle = VehicleInfo(
            vehicle_id=f"test_{i}",
            position=(0, 0),
            velocity=0,
            direction=0,
            resources={'cpu': 0.7, 'memory': 0.6, 'battery': 0.8}
        )
        vehicles.append(vehicle)
    
    # Benchmark lookup
    start_time = time.time()
    for _ in range(1000):
        manager.find_template_for_vehicles(vehicles[:5])
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 1000
    return f"Average lookup time: {avg_time*1000:.3f}ms"

def benchmark_pipeline_formation():
    """Benchmark pipeline formation performance"""
    from .vehicle_layer import PipelineFormation
    from .core.types import VehicleInfo, PipelineTemplate
    
    formation = PipelineFormation(VehicleInfo("test", (0, 0), 0, 0, {}))
    
    # Create test vehicles
    vehicles = []
    for i in range(10):
        vehicle = VehicleInfo(
            vehicle_id=f"test_{i}",
            position=(i*10, 0),
            velocity=20,
            direction=0,
            resources={'cpu': 0.7, 'memory': 0.6, 'battery': 0.8}
        )
        vehicles.append(vehicle)
    
    # Create test template
    from .core.types import TrainingConfig, ResourceClass
    training_config = TrainingConfig(epochs=2, batch_size=32, learning_rate=0.001)
    template = PipelineTemplate(
        template_id="test_template",
        resource_requirements=[ResourceClass.MEDIUM] * 5,
        expected_duration=15.0,
        communication_pattern=[(i, i+1) for i in range(4)],
        training_config=training_config
    )
    
    # Benchmark formation
    start_time = time.time()
    for _ in range(100):
        formation.initiate_pipeline_formation(template, vehicles)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    return f"Average formation time: {avg_time*1000:.3f}ms"

def benchmark_aggregation():
    """Benchmark aggregation performance"""
    from .edge_server import AsynchronousAggregator
    from .core.types import ModelUpdate
    import torch
    
    aggregator = AsynchronousAggregator()
    
    # Create test updates with consistent tensor dimensions
    updates = []
    for i in range(10):
        update = ModelUpdate(
            source_id=f"test_{i}",
            update_data=torch.randn(100),  # Smaller, consistent size
            metadata={'data_size': 100},
            training_mode="individual",
            fidelity_score=1.0  # Explicitly set as float
        )
        updates.append(update)
    
    # Benchmark aggregation
    start_time = time.time()
    for update in updates:
        try:
            aggregator.submit_update(update)
        except Exception as e:
            # Handle any errors gracefully
            pass
    end_time = time.time()
    
    # Wait for async aggregation
    time.sleep(1)
    
    total_time = end_time - start_time
    return f"Aggregation time for 10 updates: {total_time:.3f}s"

def benchmark_communication():
    """Benchmark communication performance"""
    from .vehicle_layer import CommunicationOptimizer
    import torch
    
    optimizer = CommunicationOptimizer()
    
    # Create test model update
    model_update = torch.randn(10000)
    
    # Benchmark compression
    start_time = time.time()
    for _ in range(100):
        compressed, ratio = optimizer.compress_model_update(model_update)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    return f"Average compression time: {avg_time*1000:.3f}ms, Ratio: {ratio:.2f}x"

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FHDP System")
    parser.add_argument('--version', action='version', version='FHDP 1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Edge server command
    server_parser = subparsers.add_parser('server', help='Run edge server')
    server_parser.add_argument('--port', type=int, default=8080, help='Server port')
    server_parser.add_argument('--config', help='Configuration file path')
    server_parser.add_argument('--coverage', help='Coverage area (widthxheight in meters)')
    
    # Vehicle command
    vehicle_parser = subparsers.add_parser('vehicle', help='Run vehicle')
    vehicle_parser.add_argument('vehicle_id', help='Vehicle identifier')
    vehicle_parser.add_argument('--position', help='Initial position (x,y)')
    vehicle_parser.add_argument('--velocity', type=float, default=15.0, help='Initial velocity (m/s)')
    vehicle_parser.add_argument('--direction', type=float, default=0.0, help='Initial direction (radians)')
    vehicle_parser.add_argument('--protocols', default='dsrc', help='Communication protocols (comma-separated)')
    vehicle_parser.add_argument('--simulate-movement', action='store_true', help='Simulate vehicle movement')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run simulation')
    sim_parser.add_argument('--num-vehicles', type=int, default=10, help='Number of vehicles')
    sim_parser.add_argument('--duration', type=int, default=60, help='Simulation duration (seconds)')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Execute command
    if args.command == 'server':
        run_edge_server(args)
    elif args.command == 'vehicle':
        run_vehicle(args)
    elif args.command == 'simulate':
        run_simulation(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()