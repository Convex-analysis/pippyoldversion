import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import logging
import os
import argparse
from model import ModelComponent, UnitModelPartition, DNNModel
from vehicle import Vehicle, FLADCluster
from swift_optimizer import SWIFTOptimizer
from optimizer import PipelineOptimizer
from visualizer import plot_cluster_dag, plot_pipeline_schedule
from cluster_util import load_cluster_from_xml, save_cluster_to_xml, generate_cluster_dag
from model_util import load_model_from_xml, save_model_to_xml, model_to_graphviz

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

def xml_workflow(cluster_xml_path, model_xml_path, output_dir="output", train_episodes=50, 
                 compare_baseline=True, device=None):
    """
    SWIFT optimization workflow that loads cluster and model from XML files.
    
    Args:
        cluster_xml_path: Path to the XML file defining the cluster
        model_xml_path: Path to the XML file defining the model
        output_dir: Directory to save output files
        train_episodes: Number of episodes for DQN training
        compare_baseline: Whether to compare with baseline optimizer
        device: Device to run on (None for auto-selection)
    
    Returns:
        Tuple of (swift_optimizer, comparison_results)
    """
    print("\n" + " SWIFT Pipeline Optimization - XML Workflow ".center(80, '=') + "\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load cluster from XML
    print("\nStep 1: Loading cluster from XML...")
    try:
        cluster = load_cluster_from_xml(cluster_xml_path)
        print(f"  Successfully loaded cluster '{cluster.name}' with {len(cluster.vehicles)} vehicles")
        print(f"  Dependencies: {cluster.dependencies}")
        
        # Visualize the cluster
        cluster_dag_path = os.path.join(output_dir, "loaded_cluster_dag.png")
        generate_cluster_dag(cluster, output_path=cluster_dag_path, show=False)
        print(f"  Cluster visualization saved to: {cluster_dag_path}")
    except Exception as e:
        logger.error(f"Failed to load cluster from {cluster_xml_path}: {e}")
        return None, None
    
    # Step 2: Load model from XML
    print("\nStep 2: Loading model from XML...")
    try:
        model = load_model_from_xml(model_xml_path)
        print(f"  Successfully loaded model '{model.name}' with:")
        print(f"  - {len(model.components)} components")
        print(f"  - {len(model.unit_partitions)} partitions")
        print(f"  - Total capacity: {model.total_capacity / 1e9:.2f} GB")
        print(f"  - Total FLOPs per sample: {model.total_flops_per_sample / 1e9:.2f} GFLOPs")
        
        # Visualize the model
        model_viz_path = os.path.join(output_dir, "loaded_model")
        model_graph = model_to_graphviz(model, output_path=model_viz_path, show_components=True)
        print(f"  Model visualization saved to: {model_viz_path}.png")
    except Exception as e:
        logger.error(f"Failed to load model from {model_xml_path}: {e}")
        return None, None
    
    # Step 3: Initialize SWIFT optimizer
    print("\nStep 3: Initializing SWIFT optimizer...")
    
    # Determine device to use
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    try:
        swift_optimizer = SWIFTOptimizer(
            cluster, 
            model, 
            batch_size=32,
            utilization=0.5, 
            memory_overhead=1.2,
            dqn_learning_rate=0.001,
            dqn_discount_factor=0.95,
            seed=42,  # For reproducibility
            device=device
        )
        
        # Display stability scores
        print("\nVehicle Stability Scores:")
        for v_id, score in swift_optimizer.stability_scores.items():
            print(f"  Vehicle {v_id}: {score:.4f}")
        
        # Plot stability scores
        plot_stability_scores(swift_optimizer.stability_scores)
    except Exception as e:
        logger.error(f"Failed to initialize SWIFT optimizer: {e}")
        return None, None
    
    # Step 4: Train the DQN model
    print(f"\nStep 4: Training DQN model for {train_episodes} episodes on {device}...")
    try:
        start_time = time.time()
        rewards = swift_optimizer.train_dqn_model(episodes=train_episodes)
        train_time = time.time() - start_time
        print(f"  Training completed in {train_time:.2f} seconds")
        
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
        logger.error(f"Error during DQN training: {e}")
        logger.info("Continuing with optimization using partially trained model...")
    
    # Step 5: Generate optimal pipeline using SWIFT
    print("\nStep 5: Finding optimal pipeline configurations using SWIFT algorithm...")
    try:
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
    except Exception as e:
        logger.error(f"Error during pipeline optimization: {e}")
        logger.info("Unable to generate pipelines. Check if model and cluster are compatible.")
        return swift_optimizer, None
    
    # Step 6: Compare with baseline optimizer if requested
    comparison_results = None
    if compare_baseline and pipelines:
        print("\nStep 6: Comparing with baseline optimizer...")
        try:
            comparison_results = compare_with_baseline(swift_optimizer, model, cluster)
        except Exception as e:
            logger.error(f"Error during baseline comparison: {e}")
            logger.info("Continuing without baseline comparison.")
    
    # Step 7: Present best pipeline and detailed metrics
    print("\nStep 7: Analyzing best pipeline...")
    best_pipeline = swift_optimizer.get_best_pipeline()
    
    if best_pipeline:
        best_path, best_strategy, best_time = best_pipeline
        print("\n" + " Best Pipeline ".center(80, '*'))
        print(f"  Execution path: {best_path}")
        print(f"  Execution time: {best_time:.4f} seconds")
        
        print("\n  Partition strategy:")
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
    else:
        print("\n  No valid pipelines were generated. Check compatibility between model and cluster.")
    
    # Save the trained model and optimizer state
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    swift_optimizer.save(os.path.join(model_dir, "swift_optimizer"))
    print(f"\nOptimizer state saved to {model_dir}/swift_optimizer")
    
    print("\n" + " SWIFT pipeline generation completed! ".center(80, '*'))
    return swift_optimizer, comparison_results

def main():
    """Parse arguments and run workflow."""
    parser = argparse.ArgumentParser(description="SWIFT Pipeline Optimization from XML files")
    
    parser.add_argument("--cluster", "-c", required=True,
                       help="Path to cluster XML configuration file")
    
    parser.add_argument("--model", "-m", required=True,
                       help="Path to model XML configuration file")
    
    parser.add_argument("--output", "-o", default="output",
                       help="Output directory for results and visualizations")
    
    parser.add_argument("--episodes", "-e", type=int, default=50,
                       help="Number of training episodes for DQN")
    
    parser.add_argument("--no-baseline", action="store_true",
                       help="Skip comparison with baseline optimizer")
    
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU usage even if CUDA is available")
    
    args = parser.parse_args()
    
    # Determine device based on args
    device = torch.device("cpu") if args.cpu else None
    
    # Run workflow
    xml_workflow(
        cluster_xml_path=args.cluster,
        model_xml_path=args.model,
        output_dir=args.output,
        train_episodes=args.episodes,
        compare_baseline=not args.no_baseline,
        device=device
    )

if __name__ == "__main__":
    main()
