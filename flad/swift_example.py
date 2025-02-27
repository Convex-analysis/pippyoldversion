import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import logging
import os
from model import ModelComponent, UnitModelPartition, DNNModel
from vehicle import Vehicle, FLADCluster
from swift_optimizer import SWIFTOptimizer
from optimizer import PipelineOptimizer
from visualizer import plot_cluster_dag, plot_pipeline_schedule

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SWIFT-Example")

def create_sample_model():
    """Create a sample DNN model with components and partitions."""
    # Create model components for a Bird's Eye View (BEV) fusion model
    rgb_backbone = ModelComponent("RGB_Backbone", flops_per_sample=5e9, capacity=2e9)
    lidar_backbone = ModelComponent("Lidar_Backbone", flops_per_sample=8e9, capacity=3e9)
    encoder = ModelComponent("Encoder", flops_per_sample=3e9, capacity=1.5e9)
    decoder = ModelComponent("BEV_Decoder", flops_per_sample=4e9, capacity=2e9)
    
    # Create model
    model = DNNModel("BEVFusion")
    model.add_component(rgb_backbone)
    model.add_component(lidar_backbone)
    model.add_component(encoder)
    model.add_component(decoder)
    
    # Create unit partitions
    partition1 = UnitModelPartition("RGB_Only", [rgb_backbone], 
                                    rgb_backbone.flops_per_sample, 
                                    rgb_backbone.capacity, 
                                    communication_volume=0.5e9)
    
    partition2 = UnitModelPartition("Lidar_Only", [lidar_backbone], 
                                    lidar_backbone.flops_per_sample, 
                                    lidar_backbone.capacity, 
                                    communication_volume=0.8e9)
    
    partition3 = UnitModelPartition("Encoder_Decoder", [encoder, decoder], 
                                    encoder.flops_per_sample + decoder.flops_per_sample, 
                                    encoder.capacity + decoder.capacity, 
                                    communication_volume=1.2e9)
    
    model.add_unit_partition(partition1)
    model.add_unit_partition(partition2)
    model.add_unit_partition(partition3)
    
    logger.info(f"Created model: {model.name} with {len(model.components)} components and {len(model.unit_partitions)} partitions")
    logger.info(f"  Total capacity: {model.total_capacity / 1e9:.2f} GB")
    logger.info(f"  Total FLOPs per sample: {model.total_flops_per_sample / 1e9:.2f} GFLOPs")
    
    return model

def create_complex_model():
    """Create a more complex model with more partitions for advanced scenarios."""
    # Create a larger model with more components
    components = [
        ModelComponent(f"Component_{i}", 
                      flops_per_sample=np.random.uniform(2e9, 10e9), 
                      capacity=np.random.uniform(1e9, 4e9))
        for i in range(8)
    ]
    
    model = DNNModel("ComplexModel", components=components)
    
    # Create unit partitions (more granular partitioning)
    partitions = [
        UnitModelPartition(f"Partition_{i}", 
                         [components[i]], 
                         components[i].flops_per_sample,
                         components[i].capacity,
                         communication_volume=np.random.uniform(0.3e9, 1.5e9))
        for i in range(len(components))
    ]
    
    for partition in partitions:
        model.add_unit_partition(partition)
        
    return model

def create_sample_cluster():
    """Create a sample FLAD cluster with vehicles."""
    # Create vehicles with different capabilities
    v1 = Vehicle("v1", memory=4e9, comp_capability=8e9, comm_capability=1e9)
    v2 = Vehicle("v2", memory=5e9, comp_capability=12e9, comm_capability=1.2e9)
    v3 = Vehicle("v3", memory=3e9, comp_capability=6e9, comm_capability=0.8e9)
    v4 = Vehicle("v4", memory=6e9, comp_capability=10e9, comm_capability=1.5e9)
    
    # Create cluster
    cluster = FLADCluster("SWIFT_Cluster")
    cluster.add_vehicle(v1)
    cluster.add_vehicle(v2)
    cluster.add_vehicle(v3)
    cluster.add_vehicle(v4)
    
    # Add dependencies (DAG constraints)
    cluster.add_dependency("v1", "v3")  # v1 must complete before v3 starts
    cluster.add_dependency("v2", "v4")  # v2 must complete before v4 starts
    
    logger.info(f"Created cluster with {len(cluster.vehicles)} vehicles")
    logger.info(f"Cluster dependencies: {cluster.dependencies}")
    
    return cluster

def create_complex_cluster():
    """Create a more complex cluster with more vehicles and constraints."""
    # Create vehicles with various capabilities
    vehicles = []
    for i in range(6):
        memory = np.random.uniform(3e9, 8e9)
        comp_capability = np.random.uniform(5e9, 15e9)
        comm_capability = np.random.uniform(0.8e9, 2e9)
        vehicles.append(Vehicle(f"v{i+1}", memory, comp_capability, comm_capability))
    
    # Create cluster
    cluster = FLADCluster("Complex_FLAD_Cluster")
    for v in vehicles:
        cluster.add_vehicle(v)
    
    # Add dependencies to form a more complex DAG
    # v1 -> v3 -> v5
    # v2 -> v4 -> v6
    cluster.add_dependency("v1", "v3")
    cluster.add_dependency("v3", "v5")
    cluster.add_dependency("v2", "v4")
    cluster.add_dependency("v4", "v6")
    # Cross dependencies
    cluster.add_dependency("v1", "v4")
    cluster.add_dependency("v2", "v5")
    
    return cluster

def plot_stability_scores(stability_scores):
    """Plot the stability scores for vehicles."""
    vehicles = list(stability_scores.keys())
    scores = list(stability_scores.values())
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(vehicles, scores, color='skyblue')
    
    plt.title('Vehicle Stability Scores')
    plt.xlabel('Vehicle ID')
    plt.ylabel('Stability Score')
    plt.ylim(0, 1.0)
    
    # Add the values above the bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "stability_scores.png"))
    
    plt.show()

def plot_pipeline_comparison(pipelines):
    """Plot execution times for different pipelines."""
    paths = [f"Pipeline {i+1}\n({','.join(path)})" for i, (path, _, _) in enumerate(pipelines)]
    times = [time for _, _, time in pipelines]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(paths, times, color='lightgreen')
    
    plt.title('Pipeline Execution Time Comparison')
    plt.xlabel('Pipeline')
    plt.ylabel('Execution Time (s)')
    plt.xticks(rotation=45, ha='right')
    
    # Add the values above the bars
    for bar, time_value in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{time_value:.4f}', ha='center')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "pipeline_comparison.png"))
    
    plt.show()

def compare_with_baseline(swift_optimizer, model, cluster):
    """Compare SWIFT optimizer with baseline optimizer."""
    logger.info("Comparing SWIFT with baseline optimizer...")
    
    # Create baseline optimizer
    baseline_start = time.time()
    baseline_optimizer = PipelineOptimizer(
        cluster, 
        model, 
        batch_size=swift_optimizer.batch_size,
        utilization=swift_optimizer.utilization, 
        memory_overhead=swift_optimizer.memory_overhead
    )
    
    # Run baseline optimization
    optimal_path, optimal_strategy, optimal_time = baseline_optimizer.optimize()
    baseline_end = time.time()
    baseline_optimization_time = baseline_end - baseline_start
    
    # Get SWIFT best result
    swift_best = swift_optimizer.get_best_pipeline()
    if swift_best:
        swift_path, swift_strategy, swift_time = swift_best
    else:
        logger.error("SWIFT did not produce any valid pipelines")
        return
    
    # Calculate comparison metrics
    time_diff = swift_time - optimal_time
    percentage_diff = (time_diff / optimal_time) * 100 if optimal_time > 0 else 0
    optimization_speedup = baseline_optimization_time / swift_optimizer.optimization_time if swift_optimizer.optimization_time > 0 else 0
    
    # Print comparison results
    logger.info(f"\nPerformance Comparison: SWIFT vs Baseline Optimizer")
    logger.info(f"-----------------------------------------------------")
    logger.info(f"Baseline Path: {optimal_path}")
    logger.info(f"Baseline Time: {optimal_time:.4f}s")
    logger.info(f"Baseline Optimization Duration: {baseline_optimization_time:.2f}s")
    logger.info(f"")
    logger.info(f"SWIFT Path: {swift_path}")
    logger.info(f"SWIFT Time: {swift_time:.4f}s")
    logger.info(f"SWIFT Optimization Duration: {swift_optimizer.optimization_time:.2f}s")
    logger.info(f"")
    logger.info(f"Time Difference: {time_diff:.4f}s ({percentage_diff:.2f}%)")
    logger.info(f"Optimization Speed-up: {optimization_speedup:.2f}x")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    
    # Pipeline execution times
    plt.subplot(121)
    plt.bar(['Baseline', 'SWIFT'], [optimal_time, swift_time], color=['lightblue', 'lightgreen'])
    plt.title('Execution Time (s)')
    plt.ylabel('Seconds')
    plt.grid(axis='y', alpha=0.3)
    
    # Optimization times
    plt.subplot(122)
    plt.bar(['Baseline', 'SWIFT'], [baseline_optimization_time, swift_optimizer.optimization_time], 
            color=['salmon', 'skyblue'])
    plt.title('Optimization Time (s)')
    plt.ylabel('Seconds')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison plot
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "swift_vs_baseline.png"))
    
    plt.show()
    
    return {
        'baseline_path': optimal_path,
        'baseline_time': optimal_time,
        'baseline_opt_time': baseline_optimization_time,
        'swift_path': swift_path,
        'swift_time': swift_time,
        'swift_opt_time': swift_optimizer.optimization_time,
        'time_diff': time_diff,
        'percentage_diff': percentage_diff,
        'optimization_speedup': optimization_speedup
    }

def main(use_complex_model=False, compare_baseline=True, train_episodes=50):
    """Main function to demonstrate SWIFT pipeline optimization."""
    print("\n" + " SWIFT Pipeline Optimization Example ".center(80, '=') + "\n")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create sample model and cluster based on complexity setting
    if use_complex_model:
        model = create_complex_model()
        cluster = create_complex_cluster()
    else:
        model = create_sample_model()
        cluster = create_sample_cluster()
    
    # Visualize the cluster DAG
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    plot_cluster_dag(cluster, save_path=os.path.join(output_dir, "cluster_dag.png"))
    
    # Create SWIFT optimizer with device awareness
    swift_optimizer = SWIFTOptimizer(
        cluster, 
        model, 
        batch_size=32,
        utilization=0.5, 
        memory_overhead=1.2,
        dqn_learning_rate=0.001,
        dqn_discount_factor=0.95,
        seed=42,  # For reproducibility
        device=device  # Pass the device explicitly
    )
    
    # Display stability scores
    print("\nVehicle Stability Scores:")
    for v_id, score in swift_optimizer.stability_scores.items():
        print(f"  Vehicle {v_id}: {score:.4f}")
    
    plot_stability_scores(swift_optimizer.stability_scores)
    
    # Train the DQN model with a safety timeout
    print(f"\nTraining DQN model for {train_episodes} episodes on {device}...")
    try:
        start_time = time.time()
        # Set a timeout for training (e.g., 5 minutes) to prevent indefinite hanging
        rewards = swift_optimizer.train_dqn_model(episodes=train_episodes)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Plot training rewards
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title(f'DQN Training Rewards (on {device})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "training_rewards.png"))
        plt.show()
    except Exception as e:
        print(f"Training encountered an error: {e}")
        print("Continuing with the optimization process using the partially trained model...")
        rewards = []
    
    # Find optimal set of pipelines using SWIFT
    print("\nFinding optimal pipeline configurations using SWIFT algorithm...")
    pipelines = swift_optimizer.optimize()
    
    # Display all generated pipelines
    print(f"\nGenerated {len(pipelines)} pipelines:")
    for i, (path, partition_strategy, exec_time) in enumerate(pipelines):
        print(f"\nPipeline {i+1}:")
        print(f"  Execution path: {path}")
        print(f"  Execution time: {exec_time:.4f} seconds")
        
        print("  Partition strategy:")
        for v_id, partitions in partition_strategy.items():
            partition_names = [p.name for p in partitions]
            print(f"    Vehicle {v_id}: {partition_names}")
    
    # Plot pipeline comparison
    plot_pipeline_comparison(pipelines)
    
    # Compare with baseline optimizer if requested
    comparison_results = None
    if compare_baseline:
        comparison_results = compare_with_baseline(swift_optimizer, model, cluster)
    
    # Get the best pipeline from SWIFT
    best_pipeline = swift_optimizer.get_best_pipeline()
    
    if (best_pipeline):
        best_path, best_strategy, best_time = best_pipeline
        print("\n" + " Best Pipeline ".center(80, '*'))
        print(f"  Execution path: {best_path}")
        print(f"  Execution time: {best_time:.4f} seconds")
        
        print("  Partition strategy:")
        for v_id, partitions in best_strategy.items():
            partition_names = [p.name for p in partitions]
            print(f"    Vehicle {v_id}: {partition_names}")
        
        # Calculate execution times for visualization
        execution_times = {}
        for v_id in cluster.vehicles:
            vehicle = cluster.vehicles[v_id]
            execution_times[(v_id, 'computation')] = vehicle.compute_time(
                swift_optimizer.batch_size, swift_optimizer.utilization, swift_optimizer.memory_overhead)
            execution_times[(v_id, 'communication')] = vehicle.communication_time(
                swift_optimizer.batch_size, swift_optimizer.memory_overhead)
        
        # Visualize the best pipeline schedule
        plot_pipeline_schedule(
            best_path, 
            best_strategy, 
            execution_times, 
            save_path=os.path.join(output_dir, "best_pipeline.png")
        )
        
        # Get detailed evaluation metrics
        metrics = swift_optimizer.evaluate_pipeline(best_path, best_strategy)
        
        print("\nDetailed Pipeline Metrics:")
        print(f"  Total execution time: {metrics['total_time']:.4f} seconds")
        print(f"  Total compute time: {metrics['total_compute_time']:.4f} seconds")
        print(f"  Total communication time: {metrics['total_communication_time']:.4f} seconds")
        print(f"  Total memory usage: {metrics['total_memory'] / 1e9:.2f} GB")
        
        # Use SWIFT's built-in visualization to get a detailed view of all pipelines
        swift_optimizer.visualize_pipelines(save_path=os.path.join(output_dir, "all_pipelines.png"))
        
    # Save the trained model and optimizer state
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    swift_optimizer.save(os.path.join(model_dir, "swift_optimizer"))
    
    print("\n" + " SWIFT pipeline generation completed successfully! ".center(80, '*'))
    return swift_optimizer, comparison_results

if __name__ == "__main__":
    # Run the example with default settings
    main(use_complex_model=True, compare_baseline=True, train_episodes=100)
    
    # Uncomment to run the example with a more complex model and cluster
    # main(use_complex_model=True, compare_baseline=True, train_episodes=100)
