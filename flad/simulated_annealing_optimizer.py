import numpy as np
import time
import random
import logging
import math
from collections import defaultdict
from optimizer import PipelineOptimizer
import matplotlib.pyplot as plt

logger = logging.getLogger("SimulatedAnnealingOptimizer")

class SimulatedAnnealingOptimizer:
    """
    Simulated Annealing Pipeline Optimizer for FLAD clusters.
    
    This optimizer uses simulated annealing, a probabilistic technique for finding
    global optimums in large search spaces, to discover optimal execution pipelines.
    """
    
    def __init__(self, cluster, model, batch_size=1, utilization=0.5, memory_overhead=1.0,
                initial_temp=100.0, cooling_rate=0.95, iterations=1000, seed=None):
        """
        Initialize the simulated annealing optimizer.
        
        Args:
            cluster: The FLAD cluster object
            model: The DNN model to optimize
            batch_size: Batch size for inference
            utilization: Computing resource utilization factor
            memory_overhead: Memory overhead factor
            initial_temp: Initial temperature for simulated annealing
            cooling_rate: Rate at which temperature decreases
            iterations: Number of iterations at each temperature
            seed: Random seed for reproducibility
        """
        self.cluster = cluster
        self.model = model
        self.batch_size = batch_size
        self.utilization = utilization
        self.memory_overhead = memory_overhead
        
        # Simulated annealing parameters
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        
        # Get all valid paths in the cluster DAG
        self.valid_paths = self._get_all_valid_paths()
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Store results
        self.best_solution = None
        self.best_fitness = float('inf')
        self.current_solution = None
        self.current_fitness = float('inf')
        self.optimization_time = 0
        self.temperature_history = []
        self.fitness_history = []
        
        # Initialize the baseline optimizer for evaluation
        self.baseline_optimizer = PipelineOptimizer(
            cluster, model, batch_size, utilization, memory_overhead)
    
    def _get_all_valid_paths(self):
        """Get all valid execution paths in the cluster DAG."""
        # Find sources (nodes with no incoming edges)
        sources = []
        for v_id in self.cluster.vehicles:
            has_incoming = False
            for _, targets in self.cluster.dependencies.items():
                if v_id in targets:
                    has_incoming = True
                    break
            if not has_incoming:
                sources.append(v_id)
        
        # Find sinks (nodes with no outgoing edges)
        sinks = []
        for v_id in self.cluster.vehicles:
            if v_id not in self.cluster.dependencies:
                sinks.append(v_id)
        
        # Find all paths from sources to sinks
        all_paths = []
        for source in sources:
            paths = self._find_paths(source, sinks, [])
            all_paths.extend(paths)
        
        return all_paths
    
    def _find_paths(self, current, sinks, visited):
        """Recursive DFS to find all paths from current node to any sink."""
        visited = visited + [current]
        
        # If we've reached a sink, return this path
        if current in sinks:
            return [visited]
        
        # Continue DFS
        paths = []
        next_nodes = self.cluster.dependencies.get(current, [])
        for next_node in next_nodes:
            if next_node not in visited:  # Avoid cycles
                new_paths = self._find_paths(next_node, sinks, visited)
                paths.extend(new_paths)
        
        return paths
    
    def _initialize_solution(self):
        """Initialize a random solution."""
        # Select a random valid path
        if not self.valid_paths:
            logger.warning("No valid paths available in the cluster DAG")
            return None
            
        path = random.choice(self.valid_paths)
        
        # Generate random partition assignment
        partition_strategy = {}
        
        # First, create a list of all partitions to assign
        all_partitions = list(self.model.unit_partitions)
        random.shuffle(all_partitions)
        
        # Distribute partitions among vehicles in the path
        for v_id in path:
            # Decide how many partitions to assign to this vehicle
            if not all_partitions:
                break
                
            max_partitions = min(3, len(all_partitions))
            num_partitions = random.randint(1, max_partitions)
            
            # Assign partitions
            assigned_partitions = []
            for _ in range(num_partitions):
                if all_partitions:
                    assigned_partitions.append(all_partitions.pop(0))
                    
            partition_strategy[v_id] = assigned_partitions
        
        # If any partitions remain, distribute them
        while all_partitions:
            v_id = random.choice(path)
            if v_id in partition_strategy:
                partition_strategy[v_id].append(all_partitions.pop(0))
            else:
                partition_strategy[v_id] = [all_partitions.pop(0)]
        
        return (path, partition_strategy)
    
    def _evaluate_fitness(self, solution):
        """Evaluate the fitness (execution time) of a solution."""
        if solution is None:
            return float('inf')
            
        path, partition_strategy = solution
        
        try:
            # Check if the solution is valid
            if not self._is_valid_solution(path, partition_strategy):
                return float('inf')
            
            # Calculate execution time
            total_time = 0
            for v_id in path:
                vehicle = self.cluster.vehicles[v_id]
                partitions = partition_strategy.get(v_id, [])
                
                # Skip if no partitions assigned
                if not partitions:
                    continue
                
                # Calculate total FLOPs for this vehicle's partitions
                total_flops = sum(p.flops_per_sample for p in partitions)
                
                # Calculate total memory needed
                total_memory = sum(p.capacity for p in partitions)
                
                # Check if exceeds memory capacity
                if total_memory * self.memory_overhead > vehicle.memory:
                    return float('inf')
                
                # Calculate computation time
                comp_time = total_flops / (vehicle.comp_capability * self.utilization)
                comp_time *= self.batch_size
                
                # Calculate communication time
                comm_volume = sum(p.communication_volume for p in partitions)
                comm_time = comm_volume / vehicle.comm_capability
                comm_time *= self.batch_size * self.memory_overhead
                
                # Add to total time
                total_time += comp_time + comm_time
            
            return total_time
        except Exception as e:
            logger.error(f"Error evaluating solution: {e}")
            return float('inf')
    
    def _is_valid_solution(self, path, partition_strategy):
        """Check if a solution is valid."""
        # Check if all vehicles in path are valid
        for v_id in path:
            if v_id not in self.cluster.vehicles:
                return False
        
        # Check if path respects dependencies
        for i in range(len(path) - 1):
            v1, v2 = path[i], path[i + 1]
            if v2 not in self.cluster.dependencies.get(v1, []):
                return False
        
        # Check if all partitions are covered
        all_partitions = set(self.model.unit_partitions)
        assigned_partitions = set()
        for partitions in partition_strategy.values():
            assigned_partitions.update(partitions)
        
        if assigned_partitions != all_partitions:
            return False
        
        # Check if each partition is assigned exactly once
        partition_count = defaultdict(int)
        for partitions in partition_strategy.values():
            for p in partitions:
                partition_count[p] += 1
        
        for p, count in partition_count.items():
            if count != 1:
                return False
        
        return True
    
    def _get_neighbor(self, solution):
        """Generate a neighboring solution by making a small change."""
        if solution is None:
            return self._initialize_solution()
            
        path, strategy = solution
        
        # Choose a neighbor generation method
        method = random.choice(['swap_partitions', 'change_path', 'redistribute'])
        
        if method == 'swap_partitions' and len(strategy) >= 2:
            # Swap a partition between two vehicles
            vehicles = list(strategy.keys())
            if len(vehicles) < 2:
                return solution
                
            v1, v2 = random.sample(vehicles, 2)
            
            if not strategy[v1] or not strategy[v2]:
                return solution
                
            # Select a random partition from each vehicle
            p1 = random.choice(strategy[v1])
            p2 = random.choice(strategy[v2])
            
            # Swap them
            strategy[v1].remove(p1)
            strategy[v2].remove(p2)
            strategy[v1].append(p2)
            strategy[v2].append(p1)
            
        elif method == 'change_path':
            # Try to modify the path slightly while respecting dependencies
            if len(path) <= 1 or not self.valid_paths:
                return solution
                
            # Select a different path
            new_path = random.choice(self.valid_paths)
            
            # Transfer the strategy to the new path
            new_strategy = {}
            for v_id in new_path:
                if v_id in strategy:
                    new_strategy[v_id] = strategy[v_id]
            
            # Redistribute any unassigned partitions
            assigned_partitions = set()
            for partitions in new_strategy.values():
                assigned_partitions.update(partitions)
            
            unassigned = [p for p in self.model.unit_partitions if p not in assigned_partitions]
            
            if unassigned:
                for p in unassigned:
                    v_id = random.choice(new_path)
                    if v_id in new_strategy:
                        new_strategy[v_id].append(p)
                    else:
                        new_strategy[v_id] = [p]
            
            path = new_path
            strategy = new_strategy
            
        elif method == 'redistribute':
            # Redistribute a random number of partitions
            all_partitions = []
            for v_id, partitions in strategy.items():
                all_partitions.extend(partitions)
                
            # Select partitions to redistribute
            num_to_redistribute = random.randint(1, max(1, len(all_partitions) // 3))
            to_redistribute = random.sample(all_partitions, num_to_redistribute)
            
            # Remove these partitions from their current vehicles
            new_strategy = {v_id: [p for p in partitions if p not in to_redistribute] 
                           for v_id, partitions in strategy.items()}
            
            # Redistribute them
            for p in to_redistribute:
                v_id = random.choice(path)
                if v_id in new_strategy:
                    new_strategy[v_id].append(p)
                else:
                    new_strategy[v_id] = [p]
            
            strategy = new_strategy
        
        return (path, strategy)
    
    def _acceptance_probability(self, old_cost, new_cost, temperature):
        """Calculate probability of accepting a worse solution."""
        if new_cost < old_cost:
            return 1.0
        
        return math.exp((old_cost - new_cost) / temperature)
    
    def optimize(self):
        """Run the simulated annealing algorithm to find an optimized pipeline."""
        start_time = time.time()
        
        # Initialize random solution
        self.current_solution = self._initialize_solution()
        if self.current_solution is None:
            logger.error("Failed to initialize solution")
            self.optimization_time = time.time() - start_time
            return []
            
        self.current_fitness = self._evaluate_fitness(self.current_solution)
        
        self.best_solution = self.current_solution
        self.best_fitness = self.current_fitness
        
        # Initialize temperature
        temperature = self.initial_temp
        
        # Store solutions for returning later
        solutions = []
        
        # Main simulated annealing loop
        iteration = 0
        while temperature > 0.01:  # Continue until temperature is very low
            self.temperature_history.append(temperature)
            self.fitness_history.append(self.current_fitness)
            
            if iteration % 50 == 0:
                logger.info(f"Iteration {iteration}: T = {temperature:.2f}, Best fitness = {self.best_fitness:.4f}")
                
            for _ in range(self.iterations):
                # Generate a neighbor solution
                neighbor = self._get_neighbor(self.current_solution)
                neighbor_fitness = self._evaluate_fitness(neighbor)
                
                # Decide whether to accept the neighbor
                if self._acceptance_probability(self.current_fitness, neighbor_fitness, temperature) > random.random():
                    self.current_solution = neighbor
                    self.current_fitness = neighbor_fitness
                    
                    # Update best solution if this one is better
                    if neighbor_fitness < self.best_fitness:
                        self.best_solution = neighbor
                        self.best_fitness = neighbor_fitness
                        
                        # Add to solutions list if it's valid
                        if neighbor_fitness != float('inf'):
                            solutions.append((*neighbor, neighbor_fitness))
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
        
        self.optimization_time = time.time() - start_time
        
        # Sort solutions by fitness
        solutions.sort(key=lambda x: x[2])
        
        # Return top 10 solutions or all if less than 10
        top_solutions = solutions[:10]
        
        logger.info(f"Simulated annealing completed in {self.optimization_time:.2f} seconds")
        logger.info(f"Best solution found with fitness {self.best_fitness}")
        
        return top_solutions
    
    def get_best_pipeline(self):
        """Get the best pipeline found during optimization."""
        if self.best_solution is None:
            return None
        
        path, strategy = self.best_solution
        fitness = self._evaluate_fitness(self.best_solution)
        
        return path, strategy, fitness
    
    def evaluate_pipeline(self, path, partition_strategy):
        """Evaluate a specific pipeline configuration and return detailed metrics."""
        total_time = 0
        total_compute_time = 0
        total_communication_time = 0
        total_memory = 0
        vehicle_metrics = {}
        
        for v_id in path:
            vehicle = self.cluster.vehicles[v_id]
            partitions = partition_strategy.get(v_id, [])
            
            # Skip if no partitions assigned
            if not partitions:
                vehicle_metrics[v_id] = {
                    'compute_time': 0,
                    'communication_time': 0,
                    'memory_used': 0,
                    'partitions': []
                }
                continue
            
            # Calculate total FLOPs for this vehicle's partitions
            total_flops = sum(p.flops_per_sample for p in partitions)
            
            # Calculate total memory needed
            memory_used = sum(p.capacity for p in partitions)
            total_memory += memory_used
            
            # Calculate computation time
            comp_time = total_flops / (vehicle.comp_capability * self.utilization)
            comp_time *= self.batch_size
            total_compute_time += comp_time
            
            # Calculate communication time
            comm_volume = sum(p.communication_volume for p in partitions)
            comm_time = comm_volume / vehicle.comm_capability
            comm_time *= self.batch_size * self.memory_overhead
            total_communication_time += comm_time
            
            # Store metrics for this vehicle
            vehicle_metrics[v_id] = {
                'compute_time': comp_time,
                'communication_time': comm_time,
                'memory_used': memory_used,
                'partitions': [p.name for p in partitions]
            }
            
            # Add to total time
            total_time += comp_time + comm_time
        
        return {
            'total_time': total_time,
            'total_compute_time': total_compute_time,
            'total_communication_time': total_communication_time,
            'total_memory': total_memory,
            'vehicle_metrics': vehicle_metrics,
            'is_valid': self._is_valid_solution(path, partition_strategy)
        }
    
    def plot_convergence(self, save_path=None):
        """Plot the convergence of the simulated annealing algorithm."""
        if not self.temperature_history or not self.fitness_history:
            logger.warning("No optimization history available to plot")
            return
            
        iterations = range(len(self.temperature_history))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot temperature
        ax1.plot(iterations, self.temperature_history, 'r-')
        ax1.set_ylabel('Temperature')
        ax1.set_title('Simulated Annealing Convergence')
        ax1.grid(True, alpha=0.3)
        
        # Plot fitness
        ax2.plot(iterations, self.fitness_history, 'b-')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Best Fitness')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Convergence plot saved to: {save_path}")
            
        plt.show()
    
    def visualize_pipeline(self, pipeline=None, save_path=None):
        """
        Visualize a pipeline configuration.
        
        Args:
            pipeline: Tuple of (path, strategy, time) to visualize
            save_path: Path to save the visualization
        """
        if pipeline is None:
            if self.best_solution is None:
                logger.warning("No pipeline available to visualize")
                return
                
            path, strategy = self.best_solution
            time = self.best_fitness
        else:
            path, strategy, time = pipeline
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create a timeline visualization
        y_pos = 0
        vehicle_positions = {}
        
        # Track current time for each vehicle
        current_time = defaultdict(float)
        
        # Define colors for different partitions
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.model.unit_partitions)))
        partition_colors = {p: colors[i] for i, p in enumerate(self.model.unit_partitions)}
        
        # Plot each vehicle's timeline
        for v_id in path:
            vehicle_positions[v_id] = y_pos
            vehicle = self.cluster.vehicles[v_id]
            partitions = strategy.get(v_id, [])
            
            # Calculate times
            start_time = current_time[v_id]
            for p in partitions:
                # Calculate computation time
                comp_flops = p.flops_per_sample
                comp_time = comp_flops / (vehicle.comp_capability * self.utilization) * self.batch_size
                
                # Plot computation block
                ax.barh(y_pos, comp_time, left=start_time, height=0.7, 
                        color=partition_colors[p], alpha=0.7)
                
                # Label
                if comp_time > 0.05 * time:  # Only label if block is significant
                    ax.text(start_time + comp_time/2, y_pos, p.name, 
                            ha='center', va='center', fontsize=8)
                
                start_time += comp_time
                
                # Calculate communication time
                comm_time = p.communication_volume / vehicle.comm_capability
                comm_time *= self.batch_size * self.memory_overhead
                
                # Plot communication block
                if comm_time > 0:
                    ax.barh(y_pos, comm_time, left=start_time, height=0.7, 
                            color='lightgray', hatch='///', alpha=0.5)
                    
                    # Label
                    if comm_time > 0.05 * time:  # Only label if block is significant
                        ax.text(start_time + comm_time/2, y_pos, "Comm", 
                                ha='center', va='center', fontsize=8)
                    
                    start_time += comm_time
            
            # Update current time for this vehicle
            current_time[v_id] = start_time
            
            # Draw dependencies
            if v_id in self.cluster.dependencies:
                for next_v in self.cluster.dependencies[v_id]:
                    if next_v in vehicle_positions:
                        ax.arrow(start_time, y_pos, 0, vehicle_positions[next_v] - y_pos, 
                                width=0.01, head_width=0.2, head_length=0.2, 
                                length_includes_head=True, fc='black', ec='black')
            
            y_pos += 1
        
        # Set labels
        ax.set_yticks(range(len(vehicle_positions)))
        ax.set_yticklabels(vehicle_positions.keys())
        ax.set_xlabel('Execution Time (s)')
        ax.set_title(f'Pipeline: {path}, Time={time:.4f}s')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pipeline visualization saved to: {save_path}")
        
        plt.show()
    
    def compare_with_baseline(self):
        """Compare simulated annealing solutions with baseline optimizer."""
        logger.info("Comparing simulated annealing with baseline optimizer...")
        
        # Run baseline optimization
        baseline_start = time.time()
        optimal_path, optimal_strategy, optimal_time = self.baseline_optimizer.optimize()
        baseline_time = time.time() - baseline_start
        
        # Get simulated annealing best result
        sa_best = self.get_best_pipeline()
        if sa_best is None:
            logger.error("Simulated annealing did not produce any valid pipelines")
            return None
            
        sa_path, sa_strategy, sa_time = sa_best
        
        # Calculate comparison metrics
        time_diff = sa_time - optimal_time
        percentage_diff = (time_diff / optimal_time) * 100 if optimal_time > 0 else 0
        optimization_speedup = baseline_time / self.optimization_time if self.optimization_time > 0 else 0
        
        comparison = {
            'baseline_path': optimal_path,
            'baseline_strategy': optimal_strategy,
            'baseline_time': optimal_time,
            'baseline_optimization_time': baseline_time,
            'sa_path': sa_path,
            'sa_strategy': sa_strategy,
            'sa_time': sa_time,
            'sa_optimization_time': self.optimization_time,
            'time_diff': time_diff,
            'percentage_diff': percentage_diff,
            'optimization_speedup': optimization_speedup
        }
        
        # Log results
        logger.info(f"Baseline path: {optimal_path}, time: {optimal_time:.4f}s")
        logger.info(f"SA path: {sa_path}, time: {sa_time:.4f}s")
        logger.info(f"Time difference: {time_diff:.4f}s ({percentage_diff:.2f}%)")
        logger.info(f"Optimization time speedup: {optimization_speedup:.2f}x")
        
        # Visualize comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot execution times
        methods = ['Baseline', 'Simulated Annealing']
        times = [optimal_time, sa_time]
        ax1.bar(methods, times, color=['skyblue', 'lightgreen'])
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Solution Quality Comparison')
        ax1.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(times):
            ax1.text(i, v + 0.01, f"{v:.4f}s", ha='center')
            
        # Plot optimization times
        opt_times = [baseline_time, self.optimization_time]
        ax2.bar(methods, opt_times, color=['orange', 'lightcoral'])
        ax2.set_ylabel('Optimization Time (s)')
        ax2.set_title('Optimization Performance Comparison')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(opt_times):
            ax2.text(i, v + 0.01, f"{v:.2f}s", ha='center')
            
        plt.tight_layout()
        plt.show()
        
        return comparison
    
    def run_parameter_sweep(self, initial_temps=[50, 100, 200], 
                           cooling_rates=[0.90, 0.95, 0.98],
                           iterations_per_temp=[500, 1000, 2000],
                           trials=3):
        """
        Run a parameter sweep to find optimal simulated annealing parameters.
        
        Args:
            initial_temps: List of initial temperatures to try
            cooling_rates: List of cooling rates to try
            iterations_per_temp: List of iterations per temperature to try
            trials: Number of trials for each parameter combination
            
        Returns:
            Dictionary with best parameters and results summary
        """
        logger.info("Starting parameter sweep for simulated annealing...")
        
        results = []
        best_fitness = float('inf')
        best_params = None
        
        for temp in initial_temps:
            for rate in cooling_rates:
                for iters in iterations_per_temp:
                    avg_fitness = 0
                    avg_time = 0
                    
                    logger.info(f"Testing: temp={temp}, cooling_rate={rate}, iterations={iters}")
                    
                    for trial in range(trials):
                        # Create new optimizer with these parameters
                        optimizer = SimulatedAnnealingOptimizer(
                            self.cluster,
                            self.model,
                            batch_size=self.batch_size,
                            utilization=self.utilization,
                            memory_overhead=self.memory_overhead,
                            initial_temp=temp,
                            cooling_rate=rate,
                            iterations=iters,
                            seed=42 + trial
                        )
                        
                        # Run optimization
                        optimizer.optimize()
                        
                        # Get results
                        best_solution = optimizer.get_best_pipeline()
                        if best_solution:
                            fitness = best_solution[2]
                            avg_fitness += fitness
                            avg_time += optimizer.optimization_time
                    
                    # Average over trials
                    avg_fitness /= trials
                    avg_time /= trials
                    
                    # Record results
                    param_result = {
                        'initial_temp': temp,
                        'cooling_rate': rate,
                        'iterations': iters,
                        'avg_fitness': avg_fitness,
                        'avg_time': avg_time
                    }
                    results.append(param_result)
                    
                    # Check if this is the best so far
                    if avg_fitness < best_fitness:
                        best_fitness = avg_fitness
                        best_params = param_result
                    
                    logger.info(f"  Average fitness: {avg_fitness:.4f}, time: {avg_time:.2f}s")
        
        # Log best parameters
        logger.info(f"Parameter sweep complete. Best parameters:")
        logger.info(f"  Initial temperature: {best_params['initial_temp']}")
        logger.info(f"  Cooling rate: {best_params['cooling_rate']}")
        logger.info(f"  Iterations per temperature: {best_params['iterations']}")
        logger.info(f"  Average fitness: {best_params['avg_fitness']:.4f}")
        
        return {
            'best_params': best_params,
            'all_results': results
        }
    
    def save_solution(self, path, solution=None):
        """
        Save the best solution or a specified solution to a file.
        
        Args:
            path: File path to save the solution
            solution: Optional specific solution to save, otherwise uses the best found
        """
        import json
        
        if solution is None:
            solution = self.get_best_pipeline()
            if solution is None:
                logger.error("No solution available to save")
                return False
        
        path_list, strategy_dict, fitness = solution
        
        # Convert strategy dict to serializable format
        serializable_strategy = {}
        for v_id, partitions in strategy_dict.items():
            serializable_strategy[v_id] = [p.name for p in partitions]
        
        solution_data = {
            'path': path_list,
            'strategy': serializable_strategy,
            'fitness': fitness,
            'optimization_time': self.optimization_time,
            'temperature_history_length': len(self.temperature_history),
            'model_name': self.model.name,
            'cluster_name': self.cluster.name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(solution_data, f, indent=2)
            logger.info(f"Solution saved to: {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving solution: {e}")
            return False
    
    def load_solution(self, path):
        """
        Load a solution from a file.
        
        Args:
            path: Path to the solution file
            
        Returns:
            Tuple of (path, strategy, fitness) or None if loading failed
        """
        import json
        
        try:
            with open(path, 'r') as f:
                solution_data = json.load(f)
            
            path_list = solution_data['path']
            serialized_strategy = solution_data['strategy']
            fitness = solution_data['fitness']
            
            # Convert serialized strategy back to actual partition objects
            strategy_dict = {}
            partition_dict = {p.name: p for p in self.model.unit_partitions}
            
            for v_id, partition_names in serialized_strategy.items():
                strategy_dict[v_id] = [partition_dict[name] for name in partition_names]
            
            logger.info(f"Solution loaded from: {path}")
            return path_list, strategy_dict, fitness
        except Exception as e:
            logger.error(f"Error loading solution: {e}")
            return None

