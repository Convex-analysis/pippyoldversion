import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import os
import argparse
from model import ModelComponent, UnitModelPartition, DNNModel
from vehicle import Vehicle, FLADCluster
from genetic_optimizer import GeneticPipelineOptimizer
from simulated_annealing_optimizer import SimulatedAnnealingOptimizer
from optimizer import PipelineOptimizer
from model_util import load_model_from_xml, save_model_to_xml, model_to_graphviz

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SOTA-Example")

def create_sample_model():
    """Create a sample DNN model with components and partitions."""
    # Create components for a multi-modal perception model
    rgb_backbone = ModelComponent("RGB_Backbone", flops_per_sample=5e9, capacity=2e9)
    lidar_backbone = ModelComponent("Lidar_Backbone", flops_per_sample=8e9, capacity=3e9)
    fusion = ModelComponent("Fusion_Module", flops_per_sample=3e9, capacity=1.5e9)
    detector = ModelComponent("Detection_Head", flops_per_sample=4e9, capacity=2e9)
    
    # Create model
    model = DNNModel("Multi-Modal-Perception")
    model.add_component(rgb_backbone)
    model.add_component(lidar_backbone)
    model.add_component(fusion)
    model.add_component(detector)
    
    # Create unit partitions
    partition1 = UnitModelPartition("RGB_Part", [rgb_backbone], 
                                   rgb_backbone.flops_per_sample, 
                                   rgb_backbone.capacity, 
                                   communication_volume=0.5e9)
    
    partition2 = UnitModelPartition("Lidar_Part", [lidar_backbone], 
                                   lidar_backbone.flops_per_sample, 
                                   lidar_backbone.capacity, 
                                   communication_volume=0.8e9)
    
    partition3 = UnitModelPartition("Fusion_Detection", [fusion, detector], 
                                   fusion.flops_per_sample + detector.flops_per_sample, 
                                   fusion.capacity + detector.capacity, 
                                   communication_volume=1.2e9)
    
    model.add_unit_partition(partition1)
    model.add_unit_partition(partition2)
    model.add_unit_partition(partition3)
    
    logger.info(f"Created model: {model.name} with {len(model.components)} components and {len(model.unit_partitions)} partitions")
    logger.info(f"  Total capacity: {model.total_capacity / 1e9:.2f} GB")
    logger.info(f"  Total FLOPs per sample: {model.total_flops_per_sample / 1e9:.2f} GFLOPs")
    
    return model

def create_complex_model():
    """Create a more complex model with more partitions."""
    # Create components for a complex autonomous driving model
    rgb_encoder = ModelComponent("RGB_Encoder", flops_per_sample=4e9, capacity=1.8e9)
    rgb_decoder = ModelComponent("RGB_Decoder", flops_per_sample=3e9, capacity=1.5e9)
    lidar_encoder = ModelComponent("Lidar_Encoder", flops_per_sample=6e9, capacity=2.5e9)
    lidar_decoder = ModelComponent("Lidar_Decoder", flops_per_sample=4.5e9, capacity=2e9)
    fusion = ModelComponent("Fusion_Module", flops_per_sample=2e9, capacity=1e9)
    segmentation = ModelComponent("Segmentation_Head", flops_per_sample=3e9, capacity=1.2e9)
    detection = ModelComponent("Detection_Head", flops_per_sample=2.5e9, capacity=1e9)
    planning = ModelComponent("Planning_Module", flops_per_sample=1.5e9, capacity=0.8e9)
    
    model = DNNModel("Autonomous_Driving_Stack")
    components = [rgb_encoder, rgb_decoder, lidar_encoder, lidar_decoder, 
                 fusion, segmentation, detection, planning]
    
    for component in components:
        model.add_component(component)
    
    # Create more granular partitions
    partitions = [
        UnitModelPartition("RGB_Encoder_Part", [rgb_encoder], rgb_encoder.flops_per_sample, 
                         rgb_encoder.capacity, communication_volume=0.4e9),
        
        UnitModelPartition("RGB_Decoder_Part", [rgb_decoder], rgb_decoder.flops_per_sample, 
                         rgb_decoder.capacity, communication_volume=0.6e9),
        
        UnitModelPartition("Lidar_Encoder_Part", [lidar_encoder], lidar_encoder.flops_per_sample, 
                         lidar_encoder.capacity, communication_volume=0.7e9),
        
        UnitModelPartition("Lidar_Decoder_Part", [lidar_decoder], lidar_decoder.flops_per_sample, 
                         lidar_decoder.capacity, communication_volume=0.5e9),
        
        UnitModelPartition("Fusion_Part", [fusion], fusion.flops_per_sample, 
                         fusion.capacity, communication_volume=0.8e9),
        
        UnitModelPartition("Perception_Heads", [segmentation, detection], 
                         segmentation.flops_per_sample + detection.flops_per_sample,
                         segmentation.capacity + detection.capacity, 
                         communication_volume=1.0e9),
        
        UnitModelPartition("Planning_Part", [planning], planning.flops_per_sample, 
                         planning.capacity, communication_volume=0.3e9),
    ]
    
    for partition in partitions:
        model.add_unit_partition(partition)
    
    logger.info(f"Created complex model: {model.name}")
    logger.info(f"  Components: {len(model.components)}")
    logger.info(f"  Partitions: {len(model.unit_partitions)}")
    logger.info(f"  Total capacity: {model.total_capacity / 1e9:.2f} GB")
    logger.info(f"  Total FLOPs per sample: {model.total_flops_per_sample / 1e9:.2f} GFLOPs")
    
    return model

def create_sample_cluster():
    """Create a sample FLAD cluster with vehicles."""
    # Create vehicles with different capabilities
    v1 = Vehicle("v1", memory=4e9, comp_capability=8e9, comm_capability=1e9)
    v2 = Vehicle("v2", memory=5e9, comp_capability=12e9, comm_capability=1.2e9)
    v3 = Vehicle("v3", memory=3e9, comp_capability=6e9, comm_capability=0.8e9)
    
    # Create cluster
    cluster = FLADCluster("Sample_Cluster")
    cluster.add_vehicle(v1)
    cluster.add_vehicle(v2)
    cluster.add_vehicle(v3)
    
    # Add dependencies (DAG constraints)
    cluster.add_dependency("v1", "v2")  # v1 must complete before v2 starts
    cluster.add_dependency("v2", "v3")  # v2 must complete before v3 starts
    
    logger.info(f"Created cluster with {len(cluster.vehicles)} vehicles")
    logger.info(f"Cluster dependencies: {cluster.dependencies}")
    
    return cluster

def create_complex_cluster():
    """Create a more complex cluster with more vehicles and intricate dependencies."""
    # Create vehicles with various capabilities
    vehicles = [
        Vehicle("v1", memory=6e9, comp_capability=15e9, comm_capability=1.5e9),  # Powerful lead vehicle
        Vehicle("v2", memory=4e9, comp_capability=10e9, comm_capability=1.2e9),  # Mid-range vehicle
        Vehicle("v3", memory=3e9, comp_capability=8e9, comm_capability=1.0e9),   # Standard vehicle
        Vehicle("v4", memory=5e9, comp_capability=12e9, comm_capability=1.3e9),  # Mid-high range vehicle
        Vehicle("v5", memory=2e9, comp_capability=5e9, comm_capability=0.8e9),   # Budget vehicle
        Vehicle("v6", memory=7e9, comp_capability=18e9, comm_capability=1.7e9),  # Premium vehicle
    ]
    
    # Create cluster
    cluster = FLADCluster("Complex_Cluster")
    for vehicle in vehicles:
        cluster.add_vehicle(vehicle)
    
    # Add dependencies to form a complex DAG
    # Chain: v1 -> v2 -> v5
    cluster.add_dependency("v1", "v2")
    cluster.add_dependency("v2", "v5")
    
    # Chain: v1 -> v3 -> v6
    cluster.add_dependency("v1", "v3")
    cluster.add_dependency("v3", "v6")
    
    # Chain: v1 -> v4 -> v6
    cluster.add_dependency("v1", "v4")
    cluster.add_dependency("v4", "v6")
    
    # Cross dependency
    cluster.add_dependency("v2", "v4")
    
    logger.info(f"Created complex cluster with {len(cluster.vehicles)} vehicles")
    logger.info(f"Cluster dependencies: {cluster.dependencies}")
    
    return cluster

def visualize_cluster(cluster, output_path=None):
    """Visualize the cluster DAG."""
    import graphviz
    
    dot = graphviz.Digraph(comment=f'Cluster: {cluster.name}')
    dot.attr(rankdir='LR', size='8,5')
    
    # Add vehicle nodes
    for v_id, vehicle in cluster.vehicles.items():
        mem_gb = vehicle.memory / 1e9
        comp_gflops = vehicle.comp_capability / 1e9
        comm_gbps = vehicle.comm_capability / 1e9
        
        label = f"{v_id}\nMem: {mem_gb:.1f} GB\nComp: {comp_gflops:.1f} GFLOPS\nComm: {comm_gbps:.1f} Gbps"
        dot.node(v_id, label, shape='box', style='filled', fillcolor='lightblue')
    
    # Add dependency edges
    for v_id, targets in cluster.dependencies.items():
        for target in targets:
            dot.edge(v_id, target)
    
    if output_path:
        dot.render(output_path, format='png', cleanup=True)
        logger.info(f"Cluster visualization saved to: {output_path}.png")
    
    return dot

def run_genetic_optimizer(cluster, model, output_dir="output"):
    """Run optimization using the Genetic Algorithm approach."""
    logger.info("\n" + "="*80)
    logger.info("Running Genetic Algorithm Optimizer".center(80))
    logger.info("="*80)
    
    start_time = time.time()
    
    # Create and configure the genetic optimizer
    ga_optimizer = GeneticPipelineOptimizer(
        cluster=cluster,
        model=model,
        batch_size=32,
        utilization=0.7,
        memory_overhead=1.2,
        population_size=50,
        generations=100,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elite_size=5,
        seed=42
    )
    
    # Run optimization
    logger.info("Starting genetic optimization...")
    pipelines = ga_optimizer.optimize()
    optimization_time = time.time() - start_time
    
    # Log results
    logger.info(f"Genetic optimization completed in {optimization_time:.2f} seconds")
    logger.info(f"Found {len(pipelines)} valid pipelines")
    
    if pipelines:
        best_path, best_strategy, best_time = pipelines[0]
        logger.info(f"\nBest pipeline:")
        logger.info(f"  Path: {best_path}")
        logger.info(f"  Execution time: {best_time:.4f}s")
        logger.info("  Partition assignments:")
        
        for v_id, partitions in best_strategy.items():
            logger.info(f"    Vehicle {v_id}: {[p.name for p in partitions]}")
    
        # Visualize the best pipeline
        os.makedirs(output_dir, exist_ok=True)
        ga_optimizer.visualize_pipelines(
            pipelines=[pipelines[0]], 
            save_path=os.path.join(output_dir, "genetic_best_pipeline.png")
        )
        
        # Plot convergence
        ga_optimizer.plot_convergence(
            save_path=os.path.join(output_dir, "genetic_convergence.png")
        )
        
        # Plot solution diversity
        ga_optimizer.plot_solution_diversity(
            save_path=os.path.join(output_dir, "genetic_diversity.png")
        )
    
    return ga_optimizer, pipelines

def run_simulated_annealing_optimizer(cluster, model, output_dir="output"):
    """Run optimization using the Simulated Annealing approach."""
    logger.info("\n" + "="*80)
    logger.info("Running Simulated Annealing Optimizer".center(80))
    logger.info("="*80)
    
    start_time = time.time()
    
    # Create and configure the simulated annealing optimizer
    sa_optimizer = SimulatedAnnealingOptimizer(
        cluster=cluster,
        model=model,
        batch_size=32,
        utilization=0.7,
        memory_overhead=1.2,
        initial_temp=100.0,
        cooling_rate=0.95,
        iterations=50,
        seed=42
    )
    
    # Run optimization
    logger.info("Starting simulated annealing optimization...")
    pipelines = sa_optimizer.optimize()
    optimization_time = time.time() - start_time
    
    # Log results
    logger.info(f"Simulated annealing completed in {optimization_time:.2f} seconds")
    logger.info(f"Found {len(pipelines)} valid pipelines")
    
    if pipelines:
        best_path, best_strategy, best_time = pipelines[0]
        logger.info(f"\nBest pipeline:")
        logger.info(f"  Path: {best_path}")
        logger.info(f"  Execution time: {best_time:.4f}s")
        logger.info("  Partition assignments:")
        
        for v_id, partitions in best_strategy.items():
            logger.info(f"    Vehicle {v_id}: {[p.name for p in partitions]}")
    
        # Visualize the best pipeline
        os.makedirs(output_dir, exist_ok=True)
        sa_optimizer.visualize_pipeline(
            pipeline=pipelines[0], 
            save_path=os.path.join(output_dir, "sa_best_pipeline.png")
        )
        
        # Plot convergence
        sa_optimizer.plot_convergence(
            save_path=os.path.join(output_dir, "sa_convergence.png")
        )
    
    return sa_optimizer, pipelines

def compare_baseline(cluster, model, ga_optimizer, sa_optimizer, output_dir="output"):
    """Compare the SOTA methods with the baseline optimizer."""
    logger.info("\n" + "="*80)
    logger.info("Comparing with Baseline Optimizer".center(80))
    logger.info("="*80)
    
    # Run baseline optimizer
    start_time = time.time()
    baseline_optimizer = PipelineOptimizer(
        cluster=cluster,
        model=model,
        batch_size=32,
        utilization=0.7,
        memory_overhead=1.2
    )
    
    baseline_path, baseline_strategy, baseline_time = baseline_optimizer.optimize()
    baseline_optimization_time = time.time() - start_time
    
    # Get best solutions from each optimizer
    ga_best = ga_optimizer.get_best_pipeline()
    sa_best = sa_optimizer.get_best_pipeline()
    
    if ga_best and sa_best:
        ga_path, ga_strategy, ga_time = ga_best
        sa_path, sa_strategy, sa_time = sa_best
        
        # Log comparison
        logger.info("\nSolution comparison:")
        logger.info(f"  Baseline: path={baseline_path}, time={baseline_time:.4f}s, opt_time={baseline_optimization_time:.2f}s")
        logger.info(f"  Genetic: path={ga_path}, time={ga_time:.4f}s, opt_time={ga_optimizer.optimization_time:.2f}s")
        logger.info(f"  SimAnneal: path={sa_path}, time={sa_time:.4f}s, opt_time={sa_optimizer.optimization_time:.2f}s")
        
        # Calculate improvements
        ga_improvement = ((baseline_time - ga_time) / baseline_time) * 100
        sa_improvement = ((baseline_time - sa_time) / baseline_time) * 100
        
        logger.info("\nPerformance improvements:")
        logger.info(f"  Genetic vs Baseline: {ga_improvement:.2f}% faster execution")
        logger.info(f"  SimAnneal vs Baseline: {sa_improvement:.2f}% faster execution")
        
        # Optimization time speedups
        ga_speedup = baseline_optimization_time / ga_optimizer.optimization_time
        sa_speedup = baseline_optimization_time / sa_optimizer.optimization_time
        
        logger.info("\nOptimization time comparison:")
        logger.info(f"  Baseline: {baseline_optimization_time:.2f}s")
        logger.info(f"  Genetic: {ga_optimizer.optimization_time:.2f}s ({ga_speedup:.2f}x vs baseline)")
        logger.info(f"  SimAnneal: {sa_optimizer.optimization_time:.2f}s ({sa_speedup:.2f}x vs baseline)")
        
        # Create visual comparison
        plt.figure(figsize=(12, 8))
        
        # Plot execution times
        plt.subplot(2, 1, 1)
        methods = ['Baseline', 'Genetic Algorithm', 'Simulated Annealing']
        exec_times = [baseline_time, ga_time, sa_time]
        colors = ['lightgrey', 'lightblue', 'lightgreen']
        
        bars = plt.bar(methods, exec_times, color=colors)
        plt.ylabel('Execution Time (s)')
        plt.title('Solution Quality Comparison')
        plt.grid(axis='y', alpha=0.3)
        
        # Add labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{exec_times[i]:.4f}s", ha='center')
        
        # Plot optimization times
        plt.subplot(2, 1, 2)
        opt_times = [baseline_optimization_time, ga_optimizer.optimization_time, sa_optimizer.optimization_time]
        opt_bars = plt.bar(methods, opt_times, color=['darkgrey', 'steelblue', 'forestgreen'])
        plt.ylabel('Optimization Time (s)')
        plt.title('Optimizer Performance Comparison')
        plt.grid(axis='y', alpha=0.3)
        
        # Add labels on bars
        for i, bar in enumerate(opt_bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{opt_times[i]:.2f}s", ha='center')
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "optimizer_comparison.png"), dpi=300)
        plt.show()
        
        return {
            'baseline': {
                'path': baseline_path,
                'time': baseline_time,
                'opt_time': baseline_optimization_time
            },
            'genetic': {
                'path': ga_path,
                'time': ga_time,
                'opt_time': ga_optimizer.optimization_time,
                'improvement': ga_improvement
            },
            'simulated_annealing': {
                'path': sa_path,
                'time': sa_time,
                'opt_time': sa_optimizer.optimization_time,
                'improvement': sa_improvement
            }
        }
    else:
        logger.warning("Cannot compare optimizers - some did not produce valid solutions")
        return None

def parameter_sweep_experiment(cluster, model, output_dir="output"):
    """Run parameter sweeps for both optimizers to find optimal configurations."""
    logger.info("\n" + "="*80)
    logger.info("Parameter Sweep Experiment".center(80))
    logger.info("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Genetic Algorithm parameter sweep
    logger.info("\nRunning Genetic Algorithm parameter sweep...")
    ga_optimizer = GeneticPipelineOptimizer(
        cluster=cluster,
        model=model,
        batch_size=32,
        utilization=0.7,
        memory_overhead=1.2
    )
    
    ga_sweep = ga_optimizer.run_parameter_sweep(
        mutation_rates=[0.1, 0.2, 0.3],
        crossover_rates=[0.7, 0.8, 0.9],
        population_sizes=[30, 50],
        generations=30,
        trials=2
    )
    
    # Save GA sweep results
    if ga_sweep:
        best_params = ga_sweep['best_params']
        logger.info(f"\nBest Genetic Algorithm parameters:")
        logger.info(f"  Mutation rate: {best_params['mutation_rate']}")
        logger.info(f"  Crossover rate: {best_params['crossover_rate']}")
        logger.info(f"  Population size: {best_params['population_size']}")
        logger.info(f"  Average fitness: {best_params['avg_fitness']:.4f}s")
        
        # Plot GA sweep results
        plt.figure(figsize=(10, 6))
        
        # Extract data from results
        all_results = ga_sweep['all_results']
        labels = [f"M{r['mutation_rate']}-C{r['crossover_rate']}-P{r['population_size']}" 
                 for r in all_results]
        fitnesses = [r['avg_fitness'] for r in all_results]
        times = [r['avg_time'] for r in all_results]
        
        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)
        labels = [labels[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        
        # Only show top 10 for clarity
        if len(labels) > 10:
            labels = labels[:10]
            fitnesses = fitnesses[:10]
            times = times[:10]
        
        width = 0.35
        x = np.arange(len(labels))
        
        plt.bar(x - width/2, fitnesses, width, label='Fitness (s)', color='lightblue')
        plt.bar(x + width/2, times, width, label='Optimization Time (s)', color='lightgreen')
        
        plt.xlabel('Parameter Combinations')
        plt.ylabel('Time (seconds)')
        plt.title('Genetic Algorithm Parameter Sweep Results')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ga_parameter_sweep.png"), dpi=300)
        plt.show()
    
    # Simulated Annealing parameter sweep
    logger.info("\nRunning Simulated Annealing parameter sweep...")
    sa_optimizer = SimulatedAnnealingOptimizer(
        cluster=cluster,
        model=model,
        batch_size=32,
        utilization=0.7,
        memory_overhead=1.2
    )
    
    sa_sweep = sa_optimizer.run_parameter_sweep(
        initial_temps=[50, 100, 200],
        cooling_rates=[0.9, 0.95],
        iterations_per_temp=[50, 100],
        trials=2
    )
    
    # Save SA sweep results
    if sa_sweep:
        best_params = sa_sweep['best_params']
        logger.info(f"\nBest Simulated Annealing parameters:")
        logger.info(f"  Initial temperature: {best_params['initial_temp']}")
        logger.info(f"  Cooling rate: {best_params['cooling_rate']}")
        logger.info(f"  Iterations per temperature: {best_params['iterations']}")
        logger.info(f"  Average fitness: {best_params['avg_fitness']:.4f}s")
        
        # Plot SA sweep results
        plt.figure(figsize=(10, 6))
        
        # Extract data from results
        all_results = sa_sweep['all_results']
        labels = [f"T{r['initial_temp']}-C{r['cooling_rate']}-I{r['iterations']}" 
                 for r in all_results]
        fitnesses = [r['avg_fitness'] for r in all_results]
        times = [r['avg_time'] for r in all_results]
        
        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)
        labels = [labels[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        
        # Only show top 10 for clarity
        if len(labels) > 10:
            labels = labels[:10]
            fitnesses = fitnesses[:10]
            times = times[:10]
        
        width = 0.35
        x = np.arange(len(labels))
        
        plt.bar(x - width/2, fitnesses, width, label='Fitness (s)', color='lightblue')
        plt.bar(x + width/2, times, width, label='Optimization Time (s)', color='lightgreen')
        
        plt.xlabel('Parameter Combinations')
        plt.ylabel('Time (seconds)')
        plt.title('Simulated Annealing Parameter Sweep Results')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sa_parameter_sweep.png"), dpi=300)
        plt.show()
    
    return {
        'genetic': ga_sweep,
        'simulated_annealing': sa_sweep
    }

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description='SOTA Pipeline Optimization Example')
    
    parser.add_argument('--complexity', '-c', choices=['simple', 'complex'], default='simple',
                       help='Use simple or complex model and cluster')
    
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory for results and visualizations')
    
    parser.add_argument('--run-sweep', action='store_true',
                       help='Run parameter sweep experiments (time-consuming)')
    
    args = parser.parse_args()
    
    # Create model and cluster based on complexity
    if args.complexity == 'complex':
        model = create_complex_model()
        cluster = create_complex_cluster()
        logger.info("Using complex model and cluster configuration")
    else:
        model = create_sample_model()
        cluster = create_sample_cluster()
        logger.info("Using simple model and cluster configuration")
    
    # Visualize cluster
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    visualize_cluster(cluster, os.path.join(output_dir, "cluster_dag"))
    
    # Visualize model
    model_to_graphviz(model, os.path.join(output_dir, "model_structure"))
    
    # Run genetic optimizer
    ga_optimizer, ga_pipelines = run_genetic_optimizer(cluster, model, output_dir)
    
    # Run simulated annealing optimizer
    sa_optimizer, sa_pipelines = run_simulated_annealing_optimizer(cluster, model, output_dir)
    
    # Compare with baseline
    compare_results = compare_baseline(cluster, model, ga_optimizer, sa_optimizer, output_dir)
    
    # Run parameter sweep if requested
    if args.run_sweep:
        sweep_results = parameter_sweep_experiment(cluster, model, output_dir)
        logger.info("\nParameter sweep completed. Results saved to output directory.")
    
    logger.info("\n" + "="*80)
    logger.info("SOTA Pipeline Optimization Example Completed".center(80))
    logger.info("="*80)

if __name__ == "__main__":
    main()
