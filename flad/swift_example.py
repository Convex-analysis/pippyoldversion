import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import logging
import os
import json
import datetime
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

def save_results_to_file(results, output_dir, filename="swift_results.json"):
    """
    Save optimization results to a JSON file.
    
    Args:
        results: Dictionary containing optimization results
        output_dir: Directory to save the results
        filename: Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Convert non-serializable objects to strings
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, list) and len(v) > 0 and hasattr(v[0], '__dict__'):
                    # Handle lists of objects
                    serializable_results[key][k] = [item.__class__.__name__ + ":" + item.name 
                                                   if hasattr(item, 'name') else str(item) for item in v]
                elif hasattr(v, '__dict__'):
                    # Handle objects
                    serializable_results[key][k] = v.__class__.__name__ + ":" + v.name if hasattr(v, 'name') else str(v)
                else:
                    serializable_results[key][k] = v
        elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], '__dict__'):
            serializable_results[key] = [item.__class__.__name__ + ":" + item.name 
                                        if hasattr(item, 'name') else str(item) for item in value]
        elif hasattr(value, '__dict__'):
            serializable_results[key] = value.__class__.__name__ + ":" + value.name if hasattr(value, 'name') else str(value)
        else:
            serializable_results[key] = value
    
    # Add metadata
    serializable_results['metadata'] = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    logger.info(f"Results saved to {filepath}")
    return filepath

def save_detailed_results(swift_optimizer, comparison_results, output_dir, prefix=""):
    """
    Save detailed optimization results to text and CSV files.
    
    Args:
        swift_optimizer: The SWIFT optimizer object
        comparison_results: Results from baseline comparison
        output_dir: Directory to save the results
        prefix: Optional prefix for filenames
    
    Returns:
        Dictionary with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_files = {}
    
    # 1. Save summary to text file
    summary_file = os.path.join(output_dir, f"{prefix}summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SWIFT PIPELINE OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic info
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write("\n")
        
        # Model and cluster info
        f.write(f"Model: {swift_optimizer.model.name}\n")
        f.write(f"  Components: {len(swift_optimizer.model.components)}\n")
        f.write(f"  Partitions: {len(swift_optimizer.model.unit_partitions)}\n")
        f.write(f"  Total capacity: {swift_optimizer.model.total_capacity / 1e9:.2f} GB\n")
        f.write(f"  Total FLOPs: {swift_optimizer.model.total_flops_per_sample / 1e9:.2f} GFLOPs\n\n")
        
        f.write(f"Cluster: {swift_optimizer.cluster.name}\n")
        f.write(f"  Vehicles: {len(swift_optimizer.cluster.vehicles)}\n")
        f.write(f"  Dependencies: {swift_optimizer.cluster.dependencies}\n\n")
        
        # Optimization parameters
        f.write("Optimization parameters:\n")
        f.write(f"  Batch size: {swift_optimizer.batch_size}\n")
        f.write(f"  Utilization: {swift_optimizer.utilization}\n")
        f.write(f"  Memory overhead: {swift_optimizer.memory_overhead}\n\n")
        
        # Vehicle stability scores
        f.write("Vehicle stability scores:\n")
        for v_id, score in swift_optimizer.stability_scores.items():
            f.write(f"  {v_id}: {score:.4f}\n")
        f.write("\n")
        
        # Best pipeline
        best_pipeline = swift_optimizer.get_best_pipeline()
        if best_pipeline:
            best_path, best_strategy, best_time = best_pipeline
            f.write("Best pipeline:\n")
            f.write(f"  Path: {best_path}\n")
            f.write(f"  Execution time: {best_time:.4f}s\n")
            f.write("  Partition assignments:\n")
            for v_id, partitions in best_strategy.items():
                f.write(f"    {v_id}: {[p.name for p in partitions]}\n")
            f.write("\n")
            
            # Detailed metrics
            metrics = swift_optimizer.evaluate_pipeline(best_path, best_strategy)
            f.write("Detailed metrics:\n")
            f.write(f"  Total time: {metrics['total_time']:.4f}s\n")
            f.write(f"  Compute time: {metrics['total_compute_time']:.4f}s\n")
            f.write(f"  Communication time: {metrics['total_communication_time']:.4f}s\n")
            f.write(f"  Total memory: {metrics['total_memory'] / 1e9:.2f} GB\n\n")
        
        # Comparison results
        if comparison_results:
            f.write("Comparison with baseline optimizer:\n")
            f.write(f"  Baseline path: {comparison_results['baseline_path']}\n")
            f.write(f"  Baseline time: {comparison_results['baseline_time']:.4f}s\n")
            f.write(f"  Baseline opt. time: {comparison_results['baseline_opt_time']:.2f}s\n\n")
            
            f.write(f"  SWIFT path: {comparison_results['swift_path']}\n")
            f.write(f"  SWIFT time: {comparison_results['swift_time']:.4f}s\n")
            f.write(f"  SWIFT opt. time: {comparison_results['swift_opt_time']:.2f}s\n\n")
            
            f.write(f"  Time difference: {comparison_results['time_diff']:.4f}s\n")
            f.write(f"  Percentage difference: {comparison_results['percentage_diff']:.2f}%\n")
            f.write(f"  Optimization speedup: {comparison_results['optimization_speedup']:.2f}x\n\n")
    
    results_files['summary'] = summary_file
    logger.info(f"Summary saved to {summary_file}")
    
    # 2. Save pipelines to CSV - FIXED: Use optimize() instead of non-existent get_all_pipelines()
    # Check if we have any pipelines from optimization
    try:
        # Try to get pipelines using optimize() which should return the list of all pipelines
        # or use an empty list if optimization was already done and we can't run it again
        pipelines = getattr(swift_optimizer, 'pipelines', [])
        if not pipelines:
            # If pipelines aren't stored, just report the best pipeline
            best_pipeline = swift_optimizer.get_best_pipeline()
            if best_pipeline:
                pipelines = [best_pipeline]
            else:
                pipelines = []
                
        if pipelines:
            pipelines_file = os.path.join(output_dir, f"{prefix}pipelines_{timestamp}.csv")
            with open(pipelines_file, 'w') as f:
                f.write("Pipeline,Path,Execution_Time,Partition_Strategy\n")
                for i, (path, strategy, time_val) in enumerate(pipelines):
                    strategy_str = ";".join([f"{v_id}:[{','.join([p.name for p in parts])}]" 
                                           for v_id, parts in strategy.items()])
                    f.write(f"{i+1},{','.join(path)},{time_val:.6f},\"{strategy_str}\"\n")
            
            results_files['pipelines'] = pipelines_file
            logger.info(f"Pipeline details saved to {pipelines_file}")
    except Exception as e:
        logger.warning(f"Could not save pipeline details: {e}")
    
    # 3. Save vehicle details
    vehicles_file = os.path.join(output_dir, f"{prefix}vehicles_{timestamp}.csv")
    with open(vehicles_file, 'w') as f:
        f.write("Vehicle_ID,Memory_GB,Compute_GFLOPS,Comm_Gbps,Stability_Score\n")
        for v_id, vehicle in swift_optimizer.cluster.vehicles.items():
            stability = swift_optimizer.stability_scores.get(v_id, 0)
            f.write(f"{v_id},{vehicle.memory/1e9:.2f},{vehicle.comp_capability/1e9:.2f}," +
                   f"{vehicle.comm_capability/1e9:.2f},{stability:.4f}\n")
    
    results_files['vehicles'] = vehicles_file
    logger.info(f"Vehicle details saved to {vehicles_file}")
    
    # 4. Save training rewards if available
    if hasattr(swift_optimizer, 'training_rewards') and swift_optimizer.training_rewards:
        rewards_file = os.path.join(output_dir, f"{prefix}rewards_{timestamp}.csv")
        with open(rewards_file, 'w') as f:
            f.write("Episode,Reward\n")
            for i, reward in enumerate(swift_optimizer.training_rewards):
                f.write(f"{i+1},{reward}\n")
        
        results_files['rewards'] = rewards_file
        logger.info(f"Training rewards saved to {rewards_file}")
    
    return results_files

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
    
    # Initialize results dictionary
    results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "train_episodes": train_episodes,
            "compare_baseline": compare_baseline,
            "device": str(device)
        },
        "files": {}
    }
    
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
        
        results["cluster"] = {
            "name": cluster.name,
            "vehicles": len(cluster.vehicles),
            "dependencies": cluster.dependencies
        }
        results["files"]["cluster_visualization"] = cluster_dag_path
    except Exception as e:
        logger.error(f"Failed to load cluster from {cluster_xml_path}: {e}")
        results["errors"] = [f"Failed to load cluster: {str(e)}"]
        save_results_to_file(results, output_dir)
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
        
        results["model"] = {
            "name": model.name,
            "components": len(model.components),
            "partitions": len(model.unit_partitions),
            "total_capacity_gb": model.total_capacity / 1e9,
            "total_flops_per_sample_gflops": model.total_flops_per_sample / 1e9
        }
        results["files"]["model_visualization"] = f"{model_viz_path}.png"
    except Exception as e:
        logger.error(f"Failed to load model from {model_xml_path}: {e}")
        results["errors"] = results.get("errors", []) + [f"Failed to load model: {str(e)}"]
        save_results_to_file(results, output_dir)
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
        stability_plot_path = os.path.join(output_dir, "stability_scores.png")
        plot_stability_scores(swift_optimizer.stability_scores)
        plt.savefig(stability_plot_path)
        
        results["stability_scores"] = swift_optimizer.stability_scores
        results["files"]["stability_scores_plot"] = stability_plot_path
    except Exception as e:
        logger.error(f"Failed to initialize SWIFT optimizer: {e}")
        results["errors"] = results.get("errors", []) + [f"Failed to initialize optimizer: {str(e)}"]
        save_results_to_file(results, output_dir)
        return None, None
    
    # Step 4: Train the DQN model
    print(f"\nStep 4: Training DQN model for {train_episodes} episodes on {device}...")
    try:
        start_time = time.time()
        rewards = swift_optimizer.train_dqn_model(episodes=train_episodes)
        train_time = time.time() - start_time
        print(f"  Training completed in {train_time:.2f} seconds")
        
        # Store the rewards for later saving
        swift_optimizer.training_rewards = rewards
        
        # Plot training rewards
        rewards_plot_path = os.path.join(output_dir, "training_rewards.png")
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title(f'DQN Training Rewards (on {device})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(rewards_plot_path)
        plt.show()
        
        results["training"] = {
            "episodes": train_episodes,
            "duration_seconds": train_time,
            "final_reward": rewards[-1] if rewards else None,
            "avg_reward": sum(rewards) / len(rewards) if rewards else None
        }
        results["files"]["rewards_plot"] = rewards_plot_path
    except Exception as e:
        logger.error(f"Error during DQN training: {e}")
        logger.info("Continuing with optimization using partially trained model...")
        results["errors"] = results.get("errors", []) + [f"Error during training: {str(e)}"]
    
    # Step 5: Generate optimal pipeline using SWIFT
    print("\nStep 5: Finding optimal pipeline configurations using SWIFT algorithm...")
    try:
        pipelines = swift_optimizer.optimize()
        
        # Store the pipelines for later use
        swift_optimizer.pipelines = pipelines
        
        # Display all generated pipelines
        print(f"\nGenerated {len(pipelines)} pipelines:")
        pipelines_data = []
        
        for i, (path, partition_strategy, exec_time) in enumerate(pipelines):
            print(f"\nPipeline {i+1}:")
            print(f"  Execution path: {path}")
            print(f"  Execution time: {exec_time:.4f} seconds")
            
            print("  Partition strategy:")
            strategy_data = {}
            for v_id, partitions in partition_strategy.items():
                partition_names = [p.name for p in partitions]
                print(f"    Vehicle {v_id}: {partition_names}")
                strategy_data[v_id] = partition_names
            
            pipelines_data.append({
                "path": path,
                "execution_time": exec_time,
                "strategy": strategy_data
            })
        
        # Plot pipeline comparison
        pipeline_plot_path = os.path.join(output_dir, "pipeline_comparison.png")
        plot_pipeline_comparison(pipelines)
        plt.savefig(pipeline_plot_path)
        
        results["pipelines"] = {
            "count": len(pipelines),
            "data": pipelines_data
        }
        results["files"]["pipeline_comparison_plot"] = pipeline_plot_path
    except Exception as e:
        logger.error(f"Error during pipeline optimization: {e}")
        logger.info("Unable to generate pipelines. Check if model and cluster are compatible.")
        results["errors"] = results.get("errors", []) + [f"Error during optimization: {str(e)}"]
        save_results_to_file(results, output_dir)
        return swift_optimizer, None
    
    # Step 6: Compare with baseline optimizer if requested
    comparison_results = None
    if compare_baseline and pipelines:
        print("\nStep 6: Comparing with baseline optimizer...")
        try:
            comparison_results = compare_with_baseline(swift_optimizer, model, cluster)
            
            comparison_plot_path = os.path.join(output_dir, "swift_vs_baseline.png")
            plt.savefig(comparison_plot_path)
            
            results["comparison"] = {
                "baseline_path": comparison_results["baseline_path"],
                "baseline_time": comparison_results["baseline_time"],
                "baseline_opt_time": comparison_results["baseline_opt_time"],
                "swift_path": comparison_results["swift_path"],
                "swift_time": comparison_results["swift_time"],
                "swift_opt_time": comparison_results["swift_opt_time"],
                "time_diff": comparison_results["time_diff"],
                "percentage_diff": comparison_results["percentage_diff"],
                "optimization_speedup": comparison_results["optimization_speedup"]
            }
            results["files"]["comparison_plot"] = comparison_plot_path
        except Exception as e:
            logger.error(f"Error during baseline comparison: {e}")
            logger.info("Continuing without baseline comparison.")
            results["errors"] = results.get("errors", []) + [f"Error during comparison: {str(e)}"]
    
    # Step 7: Present best pipeline and detailed metrics
    print("\nStep 7: Analyzing best pipeline...")
    best_pipeline = swift_optimizer.get_best_pipeline()
    
    if best_pipeline:
        best_path, best_strategy, best_time = best_pipeline
        print("\n" + " Best Pipeline ".center(80, '*'))
        print(f"  Execution path: {best_path}")
        print(f"  Execution time: {best_time:.4f} seconds")
        
        print("\n  Partition strategy:")
        best_strategy_data = {}
        for v_id, partitions in best_strategy.items():
            partition_names = [p.name for p in partitions]
            print(f"    Vehicle {v_id}: {partition_names}")
            best_strategy_data[v_id] = partition_names
        
        # Get detailed evaluation metrics
        metrics = swift_optimizer.evaluate_pipeline(best_path, best_strategy)
        
        print("\nDetailed Pipeline Metrics:")
        print(f"  Total execution time: {metrics['total_time']:.4f} seconds")
        print(f"  Total compute time: {metrics['total_compute_time']:.4f} seconds")
        print(f"  Total communication time: {metrics['total_communication_time']:.4f} seconds")
        print(f"  Total memory usage: {metrics['total_memory'] / 1e9:.2f} GB")
        
        # Generate pipeline visualization - FIXED: Removed 'pipeline' parameter and added 'pipelines' list
        best_pipeline_path = os.path.join(output_dir, "best_pipeline.png")
        swift_optimizer.visualize_pipelines(

            save_path=best_pipeline_path
        )
        
        results["best_pipeline"] = {
            "path": best_path,
            "execution_time": best_time,
            "strategy": best_strategy_data,
            "metrics": {
                "total_time": metrics["total_time"],
                "compute_time": metrics["total_compute_time"],
                "communication_time": metrics["total_communication_time"],
                "total_memory_gb": metrics["total_memory"] / 1e9
            }
        }
        results["files"]["best_pipeline_plot"] = best_pipeline_path
        
        # Use SWIFT's built-in visualization to get a detailed view of all pipelines
        all_pipelines_path = os.path.join(output_dir, "all_pipelines.png")
        # FIXED: For all pipelines, don't specify any parameters other than save_path
        swift_optimizer.visualize_pipelines(save_path=all_pipelines_path)
        results["files"]["all_pipelines_plot"] = all_pipelines_path
    else:
        print("\n  No valid pipelines were generated. Check compatibility between model and cluster.")
        results["errors"] = results.get("errors", []) + ["No valid pipelines were generated"]
    
    # Save the trained model and optimizer state
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    swift_model_path = os.path.join(model_dir, "swift_optimizer")
    swift_optimizer.save(swift_model_path)
    print(f"\nOptimizer state saved to {swift_model_path}")
    results["files"]["swift_model"] = swift_model_path
    
    # Save detailed results to files
    result_files = save_detailed_results(swift_optimizer, comparison_results, output_dir)
    results["files"].update({f"result_{k}": v for k, v in result_files.items()})
    
    # Save JSON results
    json_path = save_results_to_file(results, output_dir)
    
    print("\n" + " SWIFT pipeline generation completed! ".center(80, '*'))
    print(f"Full results stored in {json_path}")
    
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
