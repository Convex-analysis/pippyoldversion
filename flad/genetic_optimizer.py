import numpy as np
import time
import random
import logging
from collections import defaultdict
from optimizer import PipelineOptimizer
import matplotlib.pyplot as plt

logger = logging.getLogger("GeneticOptimizer")

class GeneticPipelineOptimizer:
    """
    Genetic Algorithm-based Pipeline Optimizer for FLAD clusters.
    
    This optimizer uses genetic algorithms to find optimal execution pipelines
    by evolving a population of potential solutions over multiple generations.
    """
    
    def __init__(self, cluster, model, batch_size=1, utilization=0.5, memory_overhead=1.0,
                population_size=50, generations=100, mutation_rate=0.2, crossover_rate=0.8, 
                elite_size=5, seed=None):
        """
        Initialize the genetic algorithm optimizer.
        
        Args:
            cluster: The FLAD cluster object
            model: The DNN model to optimize
            batch_size: Batch size for inference
            utilization: Computing resource utilization factor
            memory_overhead: Memory overhead factor
            population_size: Size of the population to evolve
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top solutions to keep unchanged
            seed: Random seed for reproducibility
        """
        self.cluster = cluster
        self.model = model
        self.batch_size = batch_size
        self.utilization = utilization
        self.memory_overhead = memory_overhead
        
        # Genetic algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Get all valid paths in the cluster DAG
        self.valid_paths = self._get_all_valid_paths()
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Store results
        self.best_solution = None
        self.best_fitness = float('inf')
        self.optimization_time = 0
        self.population_history = []
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
    
    def _initialize_population(self):
        """Initialize a random population of solutions."""
        population = []
        
        for _ in range(self.population_size):
            # Select a random valid path
            if not self.valid_paths:
                logger.warning("No valid paths available in the cluster DAG")
                return []
                
            path = random.choice(self.valid_paths)
            
            # Generate random partition assignment
            partition_strategy = {}
            for v_id in path:
                # Randomly assign partitions to this vehicle
                available_partitions = self.model.unit_partitions
                num_partitions = random.randint(1, min(3, len(available_partitions)))
                assigned_partitions = random.sample(available_partitions, num_partitions)
                partition_strategy[v_id] = assigned_partitions
            
            # Add to population
            population.append((path, partition_strategy))
        
        return population
    
    def _evaluate_fitness(self, solution):
        """Evaluate the fitness (execution time) of a solution."""
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
                total_flops = sum(p.flops for p in partitions)
                
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
        
        # Check if vehicle memory constraints are satisfied
        for v_id, partitions in partition_strategy.items():
            if v_id not in self.cluster.vehicles:
                return False
                
            vehicle = self.cluster.vehicles[v_id]
            total_memory = sum(p.capacity for p in partitions)
            if total_memory * self.memory_overhead > vehicle.memory:
                return False
        
        return True
    
    def _select_parent(self, population, fitnesses):
        """Select a parent using tournament selection."""
        # Tournament selection
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        # Select the best from the tournament
        best_idx = tournament_indices[np.argmin(tournament_fitnesses)]
        return population[best_idx]
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1
        
        path1, strategy1 = parent1
        path2, strategy2 = parent2
        
        # Create a new path by selecting a random crossover point
        if len(path1) > 1 and len(path2) > 1:
            crossover_point1 = random.randint(1, len(path1) - 1)
            crossover_point2 = random.randint(1, len(path2) - 1)
            
            new_path = path1[:crossover_point1] + path2[crossover_point2:]
            
            # Check if the new path is valid, if not, use parent1's path
            for i in range(len(new_path) - 1):
                v1, v2 = new_path[i], new_path[i + 1]
                if v2 not in self.cluster.dependencies.get(v1, []):
                    new_path = path1
                    break
        else:
            new_path = path1
        
        # Create a new partition strategy by mixing both parents
        new_strategy = {}
        for v_id in new_path:
            if v_id in strategy1 and v_id in strategy2:
                # Choose strategy from either parent
                if random.random() < 0.5:
                    new_strategy[v_id] = strategy1[v_id]
                else:
                    new_strategy[v_id] = strategy2[v_id]
            elif v_id in strategy1:
                new_strategy[v_id] = strategy1[v_id]
            elif v_id in strategy2:
                new_strategy[v_id] = strategy2[v_id]
        
        return (new_path, new_strategy)
    
    def _mutate(self, solution):
        """Apply mutation to a solution."""
        if random.random() > self.mutation_rate:
            return solution
        
        path, strategy = solution
        
        # Mutation type 1: Swap two adjacent vehicles in path
        if len(path) > 1 and random.random() < 0.3:
            i = random.randint(0, len(path) - 2)
            # Check if swap is valid based on dependencies
            v1, v2 = path[i], path[i + 1]
            if v1 in self.cluster.dependencies.get(v2, []):  # If v2 can precede v1
                new_path = path.copy()
                new_path[i], new_path[i + 1] = new_path[i + 1], new_path[i]
                path = new_path
        
        # Mutation type 2: Reassign a partition to a different vehicle
        if random.random() < 0.7:
            all_partitions = list(self.model.unit_partitions)
            if all_partitions:
                # Select a random partition
                partition = random.choice(all_partitions)
                
                # Find current vehicle with this partition
                current_vehicle = None
                for v_id, partitions in strategy.items():
                    if partition in partitions:
                        current_vehicle = v_id
                        break
                
                # If found, move to another vehicle
                if current_vehicle is not None:
                    # Select a different vehicle
                    available_vehicles = [v for v in path if v != current_vehicle]
                    if available_vehicles:
                        target_vehicle = random.choice(available_vehicles)
                        
                        # Remove from current vehicle
                        strategy[current_vehicle].remove(partition)
                        
                        # Add to target vehicle
                        if target_vehicle in strategy:
                            strategy[target_vehicle].append(partition)
                        else:
                            strategy[target_vehicle] = [partition]
        
        return (path, strategy)
    
    def optimize(self):
        """Run the genetic algorithm to find an optimized pipeline."""
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population()
        if not population:
            logger.error("Failed to initialize population")
            self.optimization_time = time.time() - start_time
            return []
        
        # Main evolution loop
        for generation in range(self.generations):
            # Evaluate fitness for all solutions
            fitnesses = [self._evaluate_fitness(sol) for sol in population]
            
            # Keep track of best solution
            min_fitness_idx = np.argmin(fitnesses)
            min_fitness = fitnesses[min_fitness_idx]
            
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_solution = population[min_fitness_idx]
            
            # Store history for analysis
            avg_fitness = np.mean([f for f in fitnesses if f != float('inf')])
            self.population_history.append(population)
            self.fitness_history.append((min_fitness, avg_fitness))
            
            # Logging
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {min_fitness}, Avg fitness = {avg_fitness}")
            
            # Create the next generation
            next_generation = []
            
            # Elitism: keep top solutions unchanged
            sorted_indices = np.argsort(fitnesses)
            for i in range(self.elite_size):
                if i < len(sorted_indices):
                    next_generation.append(population[sorted_indices[i]])
            
            # Create offspring for the rest of the population
            while len(next_generation) < self.population_size:
                parent1 = self._select_parent(population, fitnesses)
                parent2 = self._select_parent(population, fitnesses)
                
                offspring = self._crossover(parent1, parent2)
                offspring = self._mutate(offspring)
                
                next_generation.append(offspring)
            
            population = next_generation
        
        self.optimization_time = time.time() - start_time
        
        # Return the top N solutions
        final_fitnesses = [self._evaluate_fitness(sol) for sol in population]
        sorted_indices = np.argsort(final_fitnesses)
        
        top_solutions = []
        for i in sorted_indices[:10]:  # Return top 10 solutions
            if final_fitnesses[i] != float('inf'):
                solution = population[i]
                top_solutions.append((*solution, final_fitnesses[i]))
        
        logger.info(f"Genetic algorithm optimization completed in {self.optimization_time:.2f} seconds")
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
        
        for v_id in path:
            vehicle = self.cluster.vehicles[v_id]
            partitions = partition_strategy.get(v_id, [])
            
            # Skip if no partitions assigned
            if not partitions:
                continue
            
            # Calculate total FLOPs for this vehicle's partitions
            total_flops = sum(p.flops for p in partitions)
            
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
            
            # Add to total time
            total_time += comp_time + comm_time
        
        return {
            'total_time': total_time,
            'total_compute_time': total_compute_time,
            'total_communication_time': total_communication_time,
            'total_memory': total_memory,
            'is_valid': self._is_valid_solution(path, partition_strategy)
        }
    
    def plot_convergence(self, save_path=None):
        """Plot the convergence of the genetic algorithm."""
        import matplotlib.pyplot as plt
        
        generations = range(len(self.fitness_history))
        best_fitness = [f[0] for f in self.fitness_history]
        avg_fitness = [f[1] for f in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness')
        plt.plot(generations, avg_fitness, 'r-', label='Average Fitness')
        plt.title('Genetic Algorithm Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Execution Time)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def visualize_pipelines(self, pipelines=None, save_path=None):
        """
        Visualize the top pipeline configurations.
        
        Args:
            pipelines: List of (path, strategy, time) tuples to visualize
            save_path: Path to save the visualization
        """
        if pipelines is None:
            # Use the final generation results
            if not self.population_history or not self.fitness_history:
                logger.warning("No optimization history available to visualize")
                return
            
            # Get last generation and its fitness
            last_population = self.population_history[-1]
            fitnesses = [self._evaluate_fitness(sol) for sol in last_population]
            
            # Sort by fitness
            sorted_indices = np.argsort(fitnesses)
            pipelines = [(last_population[i][0], last_population[i][1], fitnesses[i]) 
                         for i in sorted_indices[:5] if fitnesses[i] != float('inf')]
        
        if not pipelines:
            logger.warning("No valid pipelines to visualize")
            return
        
        fig, axes = plt.subplots(len(pipelines), 1, figsize=(12, 4 * len(pipelines)))
        if len(pipelines) == 1:
            axes = [axes]
        
        for i, (path, strategy, execution_time) in enumerate(pipelines):
            ax = axes[i]
            
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
                    if comp_time > 0.05 * execution_time:  # Only label if block is significant
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
                        if comm_time > 0.05 * execution_time:  # Only label if block is significant
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
            ax.set_title(f'Pipeline {i+1}: Path={path}, Time={execution_time:.4f}s')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pipeline visualization saved to: {save_path}")
        
        plt.show()
    
    def compare_with_baseline(self):
        """Compare genetic algorithm solutions with baseline optimizer."""
        logger.info("Comparing genetic algorithm with baseline optimizer...")
        
        # Run baseline optimization
        baseline_start = time.time()
        optimal_path, optimal_strategy, optimal_time = self.baseline_optimizer.optimize()
        baseline_time = time.time() - baseline_start
        
        # Get genetic algorithm best result
        ga_best = self.get_best_pipeline()
        if ga_best is None:
            logger.error("Genetic algorithm did not produce any valid pipelines")
            return None
            
        ga_path, ga_strategy, ga_time = ga_best
        
        # Calculate comparison metrics
        time_diff = ga_time - optimal_time
        percentage_diff = (time_diff / optimal_time) * 100 if optimal_time > 0 else 0
        optimization_speedup = baseline_time / self.optimization_time if self.optimization_time > 0 else 0
        
        comparison = {
            'baseline_path': optimal_path,
            'baseline_strategy': optimal_strategy,
            'baseline_time': optimal_time,
            'baseline_optimization_time': baseline_time,
            'ga_path': ga_path,
            'ga_strategy': ga_strategy,
            'ga_time': ga_time,
            'ga_optimization_time': self.optimization_time,
            'time_diff': time_diff,
            'percentage_diff': percentage_diff,
            'optimization_speedup': optimization_speedup
        }
        
        # Log results
        logger.info(f"Baseline path: {optimal_path}, time: {optimal_time:.4f}s")
        logger.info(f"GA path: {ga_path}, time: {ga_time:.4f}s")
        logger.info(f"Time difference: {time_diff:.4f}s ({percentage_diff:.2f}%)")
        logger.info(f"Optimization time speedup: {optimization_speedup:.2f}x")
        
        return comparison
    
    def plot_solution_diversity(self, save_path=None):
        """Plot the diversity of solutions in the population over generations."""
        if not self.population_history:
            logger.warning("No optimization history available for diversity analysis")
            return
        
        # Calculate diversity measure for each generation
        diversity = []
        for generation in self.population_history:
            # Count unique paths
            unique_paths = set(tuple(sol[0]) for sol in generation)
            
            # Calculate average partition overlap
            overlap_sum = 0
            overlap_count = 0
            
            for i in range(len(generation)):
                for j in range(i+1, len(generation)):
                    # Count shared partitions between solutions i and j
                    strategy_i = generation[i][1]
                    strategy_j = generation[j][1]
                    
                    # Calculate overlap as Jaccard similarity
                    i_assignments = {(v, tuple(sorted(p.name for p in partitions))) 
                                    for v, partitions in strategy_i.items()}
                    j_assignments = {(v, tuple(sorted(p.name for p in partitions))) 
                                    for v, partitions in strategy_j.items()}
                    
                    intersection = len(i_assignments.intersection(j_assignments))
                    union = len(i_assignments.union(j_assignments))
                    
                    overlap = intersection / union if union > 0 else 0
                    overlap_sum += overlap
                    overlap_count += 1
            
            avg_overlap = overlap_sum / overlap_count if overlap_count > 0 else 0
            diversity_score = (len(unique_paths) / len(generation)) * (1 - avg_overlap)
            diversity.append(diversity_score)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(diversity)), diversity, 'g-')
        plt.title('Population Diversity Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Diversity Score (Higher = More Diverse)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Diversity plot saved to: {save_path}")
        
        plt.show()
    
    def run_parameter_sweep(self, mutation_rates=[0.1, 0.2, 0.3], 
                           crossover_rates=[0.7, 0.8, 0.9],
                           population_sizes=[30, 50, 70],
                           generations=50, trials=3):
        """
        Run a parameter sweep to find optimal genetic algorithm parameters.
        
        Args:
            mutation_rates: List of mutation rates to try
            crossover_rates: List of crossover rates to try
            population_sizes: List of population sizes to try
            generations: Number of generations for each trial
            trials: Number of trials for each parameter combination
            
        Returns:
            Dictionary with best parameters and results summary
        """
        logger.info("Starting parameter sweep for genetic algorithm...")
        
        results = []
        best_fitness = float('inf')
        best_params = None
        
        for mut_rate in mutation_rates:
            for cross_rate in crossover_rates:
                for pop_size in population_sizes:
                    avg_fitness = 0
                    avg_time = 0
                    
                    logger.info(f"Testing: mutation={mut_rate}, crossover={cross_rate}, population={pop_size}")
                    
                    for trial in range(trials):
                        # Create new optimizer with these parameters
                        optimizer = GeneticPipelineOptimizer(
                            self.cluster,
                            self.model,
                            batch_size=self.batch_size,
                            utilization=self.utilization,
                            memory_overhead=self.memory_overhead,
                            population_size=pop_size,
                            generations=generations,
                            mutation_rate=mut_rate,
                            crossover_rate=cross_rate,
                            elite_size=max(1, int(pop_size * 0.1)),
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
                        'mutation_rate': mut_rate,
                        'crossover_rate': cross_rate,
                        'population_size': pop_size,
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
        logger.info(f"  Mutation rate: {best_params['mutation_rate']}")
        logger.info(f"  Crossover rate: {best_params['crossover_rate']}")
        logger.info(f"  Population size: {best_params['population_size']}")
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
            'generation_count': len(self.fitness_history),
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
            logger.info(f"  Path: {path_list}")
            logger.info(f"  Fitness: {fitness}")
            
            return path_list, strategy_dict, fitness
        except Exception as e:
            logger.error(f"Error loading solution: {e}")
            return None
