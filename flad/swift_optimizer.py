import numpy as np
import networkx as nx
from collections import defaultdict
import torch
import time
from typing import Dict, List, Tuple, Set, Optional
import logging

from model import DNNModel
from vehicle import FLADCluster, Vehicle
from dqn_model import DQNPipelineModel
from tqdm import tqdm  # Ensure tqdm is imported

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SWIFT")

class StabilityScorer:
    """Calculate stability scores for vehicles in a FLAD cluster."""
    
    def __init__(self, cluster, seed=None, history_weight=0.7):
        """
        Args:
            cluster: FLADCluster object
            seed: Random seed for reproducibility
            history_weight: Weight given to historical data vs. capability heuristics
        """
        self.cluster = cluster
        self.random_gen = np.random.RandomState(seed)
        self.history_weight = history_weight
        
    def calculate_scores(self, history=None):
        """
        Calculate stability scores for each vehicle.
        
        Args:
            history: Optional historical data about vehicle presence in cluster
            
        Returns:
            Dict mapping vehicle ID to stability score (0-1 range)
        """
        if history:
            # If we have historical data, combine it with capability heuristics
            history_scores = self._calculate_from_history(history)
            heuristic_scores = self._calculate_heuristic()
            
            # Weighted combination of history and heuristics
            scores = {}
            for v_id in self.cluster.vehicles:
                scores[v_id] = (self.history_weight * history_scores[v_id] + 
                               (1 - self.history_weight) * heuristic_scores[v_id])
            return scores
        else:
            # Generate random stability scores as attributes of vehicles
            return self._generate_random_scores()
    
    def _calculate_from_history(self, history):
        """Calculate stability scores from historical presence data."""
        scores = {}
        total_samples = len(history)
        
        # Default scores based on capabilities if vehicle has no history
        default_scores = self._calculate_heuristic()
        
        for v_id, vehicle in self.cluster.vehicles.items():
            if v_id in history:
                # Calculate percentage of time vehicle was present in cluster
                presence_count = sum(v_id in sample for sample in history)
                scores[v_id] = presence_count / total_samples
            else:
                # Use capability-based default if no history
                scores[v_id] = default_scores[v_id]
            
        return scores
    
    def _calculate_heuristic(self):
        """Calculate stability scores using heuristics based on vehicle capabilities."""
        scores = {}
        
        # Get maximum values for normalization
        max_memory = max(v.memory for v in self.cluster.vehicles.values())
        max_comp = max(v.comp_capability for v in self.cluster.vehicles.values())
        max_comm = max(v.comm_capability for v in self.cluster.vehicles.values())
        
        for v_id, vehicle in self.cluster.vehicles.items():
            # Normalize capabilities to 0-1 range
            memory_factor = vehicle.memory / max_memory
            comp_factor = vehicle.comp_capability / max_comp
            comm_factor = vehicle.comm_capability / max_comm
            
            # Combined score with weights
            # Memory gets higher weight as it's often the limiting factor in FLAD
            scores[v_id] = 0.5 * memory_factor + 0.3 * comp_factor + 0.2 * comm_factor
            
        return scores
    
    def _generate_random_scores(self):
        """Generate random stability scores for vehicles."""
        scores = {}
        
        for v_id in self.cluster.vehicles:
            # Generate a random stability score between 0.2 and 0.95
            # Higher scores mean more stable vehicles
            scores[v_id] = 0.2 + 0.75 * self.random_gen.random()
            
        return scores


class SWIFTOptimizer:
    """
    SWIFT (Speedy Weight-based Intelligent Fast Two-phase scheduler) optimizer
    for pipeline generation in FLAD clusters.
    """
    
    def __init__(self, cluster: FLADCluster, model: DNNModel, batch_size: int = 1, 
                 utilization: float = 0.5, memory_overhead: float = 1.2, 
                 dqn_learning_rate: float = 0.001, dqn_discount_factor: float = 0.95,
                 stability_history=None, seed=None, device=None):
        """
        Args:
            cluster: FLADCluster object
            model: DNNModel object
            batch_size: Number of samples per batch
            utilization: GPU utilization factor (between 0.3 and 0.7)
            memory_overhead: Memory bandwidth overhead factor (between 1.1 and 1.5)
            dqn_learning_rate: Learning rate for DQN model
            dqn_discount_factor: Discount factor for DQN model
            stability_history: Optional history data for stability scoring
            seed: Random seed for reproducibility
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.cluster = cluster
        self.model = model
        self.batch_size = batch_size
        self.utilization = utilization
        self.memory_overhead = memory_overhead
        self.seed = seed
        
        # Set device (default to CUDA if available, otherwise CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Calculate stability scores - crucial for SWIFT's two-phase approach
        self.stability_scorer = StabilityScorer(cluster, seed=seed)
        self.stability_scores = self.stability_scorer.calculate_scores(stability_history)
        logger.info(f"Calculated stability scores: {self.stability_scores}")
        
        # Initialize DQN model with device awareness
        state_size = self._calculate_state_size()
        action_size = self._calculate_action_size()
        logger.info(f"DQN model state size: {state_size}, action size: {action_size}, device: {self.device}")
        
        self.dqn_model = DQNPipelineModel(
            state_size=state_size,
            action_size=action_size,
            learning_rate=dqn_learning_rate,
            discount_factor=dqn_discount_factor,
            prioritized_replay=True,  # Use prioritized replay for better sample efficiency
            double_dqn=True,          # Use double DQN to reduce overestimation bias
            dueling_network=True      # Use dueling network to better estimate state values
        )
        
        # Move DQN model to appropriate device
        self.dqn_model.device = self.device
        self.dqn_model.q_network.to(self.device)
        self.dqn_model.target_network.to(self.device)
        
        # Store pipeline configurations
        self.pipelines = []  # List of (path, partition_strategy, execution_time) tuples
        
        # Performance metrics
        self.optimization_time = 0
    
    def _calculate_state_size(self):
        """
        Calculate the state size for DQN based on cluster and model.
        
        The state space includes:
        1. Available model capacity (1 value)
        2. Current model partitions (1 value per vehicle)
        3. Memory efficiency ratios (1 value per vehicle)
        4. Computation and communication times (2 values per vehicle)
        5. Execution path representation (binary adjacency matrix flattened)
        """
        n_vehicles = len(self.cluster.vehicles)
        return 1 + 4 * n_vehicles + (n_vehicles * n_vehicles)
    
    def _calculate_action_size(self):
        """
        Calculate the action size for DQN.
        
        The action space combines partition assignment and scheduling:
        - Which vehicle to add to the path next (vehicle selection)
        - How much model capacity to assign to that vehicle (partition size)
        """
        # Action space is (number of vehicles) × (partition size levels)
        # For simplicity, discretize partition sizes into 5 levels per vehicle
        return len(self.cluster.vehicles) * 5
    
    def generate_state(self, current_partitions: Dict[str, List], current_path=None):
        """
        Generate state representation for DQN.
        
        Args:
            current_partitions: Dict mapping vehicle ID to current assigned partitions
            current_path: List of vehicle IDs representing current execution path
            
        Returns:
            Numpy array representing the state
        """
        n_vehicles = len(self.cluster.vehicles)
        state = []
        
        # 1. Available model capacity
        total_assigned = sum(sum(p.capacity for p in partitions) 
                             for partitions in current_partitions.values())
        available_capacity = self.model.total_capacity - total_assigned
        state.append(available_capacity / self.model.total_capacity)  # Normalize
        
        vehicle_ids = list(self.cluster.vehicles.keys())
        
        # 2-4. Vehicle-specific states
        for v_id in vehicle_ids:
            vehicle = self.cluster.vehicles[v_id]
            partitions = current_partitions.get(v_id, [])
            
            # 2. Current model partitions
            current_partition_size = sum(p.capacity for p in partitions)
            state.append(current_partition_size / self.model.total_capacity)
            
            # 3. Memory efficiency ratios
            if vehicle.memory > 0:
                memory_efficiency = current_partition_size / vehicle.memory
            else:
                memory_efficiency = 0
            state.append(memory_efficiency)
            
            # 4. Computation and communication times
            vehicle.assigned_partitions = partitions
            compute_time = vehicle.compute_time(self.batch_size, self.utilization, self.memory_overhead)
            comm_time = vehicle.communication_time(self.batch_size, self.memory_overhead)
            
            # Normalize times (approximation)
            state.append(compute_time / 10.0)  # Assuming max compute time is around 10s
            state.append(comm_time / 10.0)     # Assuming max comm time is around 10s
        
        # 5. Execution path representation as a flattened adjacency matrix
        adjacency_matrix = np.zeros((n_vehicles, n_vehicles))
        if current_path:
            id_to_idx = {v_id: i for i, v_id in enumerate(vehicle_ids)}
            for i in range(len(current_path) - 1):
                src_idx = id_to_idx[current_path[i]]
                dst_idx = id_to_idx[current_path[i+1]]
                adjacency_matrix[src_idx][dst_idx] = 1
                
        state.extend(adjacency_matrix.flatten())
        
        return np.array(state, dtype=np.float32)
    
    def calculate_reward(self, path: List[str], partition_strategy: Dict[str, List]):
        """
        Calculate reward for a given pipeline configuration based on multiple factors.
        
        Args:
            path: List of vehicle IDs representing execution order
            partition_strategy: Dict mapping vehicle ID to list of assigned partitions
            
        Returns:
            Reward value combining performance and constraint satisfaction
        """
        # Initialize base reward
        reward = 0
        
        # Update vehicle assignments to calculate performance metrics
        for v_id, partitions in partition_strategy.items():
            self.cluster.vehicles[v_id].assigned_partitions = partitions
        
        # --- Constraint Satisfaction Component ---
        
        # Check memory constraints
        memory_constraint_violated = False
        for v_id, vehicle in self.cluster.vehicles.items():
            partitions = partition_strategy.get(v_id, [])
            total_size = sum(p.capacity for p in partitions)
            if total_size > vehicle.memory:
                reward -= 100  # Heavy penalty for exceeding memory constraints
                memory_constraint_violated = True
        
        # Check for overlapping partitions (each partition should be assigned only once)
        partition_violation = False
        assigned_partition_ids = []
        for partitions in partition_strategy.values():
            for p in partitions:
                if id(p) in assigned_partition_ids:
                    reward -= 100  # Heavy penalty for duplicate assignment
                    partition_violation = True
                else:
                    assigned_partition_ids.append(id(p))
        
        # Check DAG precedence constraints
        dag_violation = False
        if not self.cluster.is_valid_path(path):
            reward -= 100  # Heavy penalty for invalid path
            dag_violation = True
        
        # If any hard constraint is violated, return the penalty immediately
        if memory_constraint_violated or partition_violation or dag_violation:
            return reward
        
        # --- Performance Optimization Component ---
        
        # Calculate execution time factors
        total_time = 0
        total_compute_time = 0
        total_comm_time = 0
        
        for i, v_id in enumerate(path):
            vehicle = self.cluster.vehicles[v_id]
            
            # Compute time factors
            compute_time = vehicle.compute_time(
                self.batch_size, self.utilization, self.memory_overhead)
            total_compute_time += compute_time
            total_time += compute_time
            
            # Penalize compute time - More expensive for less stable vehicles
            stability_factor = self.stability_scores.get(v_id, 0.5)
            compute_penalty = compute_time * (2 - stability_factor)  # Higher penalty for less stable vehicles
            reward -= compute_penalty * 5  # Scale factor
            
            # Communication time factors (except for last vehicle)
            if i < len(path) - 1:
                comm_time = vehicle.communication_time(self.batch_size, self.memory_overhead)
                total_comm_time += comm_time
                total_time += comm_time
                reward -= comm_time * 5  # Scale factor for communication time
        
        # Terminal reward based on total path execution time
        reward -= total_time * 10  # Scale factor for total time
        
        # --- Resource Utilization Component ---
        
        # Reward efficient memory utilization across vehicles
        assigned_capacity = sum(sum(p.capacity for p in partitions) 
                              for partitions in partition_strategy.values())
        # Penalize unused capacity
        if assigned_capacity < self.model.total_capacity:
            utilization_ratio = assigned_capacity / self.model.total_capacity
            reward -= (1 - utilization_ratio) * 30  # Scale factor for unused capacity
        
        # Reward balanced memory utilization across vehicles
        memory_utilization = []
        for v_id, vehicle in self.cluster.vehicles.items():
            partitions = partition_strategy.get(v_id, [])
            if vehicle.memory > 0 and partitions:
                memory_utilization.append(sum(p.capacity for p in partitions) / vehicle.memory)
        
        if memory_utilization:
            # Standard deviation of memory utilization - lower is better (more balanced)
            balance_metric = np.std(memory_utilization) if len(memory_utilization) > 1 else 0
            reward -= balance_metric * 20  # Scale factor for balance
        
        return reward
    
    def initial_pipeline_greedy(self):
        """
        Generate initial pipeline using greedy approach based on stability scores.
        
        This implements the first phase of SWIFT algorithm, establishing a quick
        initial pipeline by prioritizing vehicles with higher stability scores.
        
        Returns:
            Tuple of (path, partition_strategy)
        """
        logger.info("Generating initial pipeline using stability-based greedy approach")
        start_time = time.time()
        
        # Sort vehicles by stability scores (descending)
        sorted_vehicles = sorted(
            self.cluster.vehicles.values(),
            key=lambda v: self.stability_scores[v.id],
            reverse=True
        )
        
        # Initialize path with vehicles in order of stability
        path = [v.id for v in sorted_vehicles]
        
        # Check if path respects DAG constraints
        if not self.cluster.is_valid_path(path):
            logger.info("Initial path violates DAG constraints, performing topological sort")
            # If not, perform topological sort while trying to maintain stability order
            G = nx.DiGraph()
            for v_id in self.cluster.vehicles:
                G.add_node(v_id, stability=self.stability_scores[v_id])
            
            for source, target in self.cluster.dependencies:
                G.add_edge(source, target)
            
            # Lexicographical topological sort that respects both dependencies and stability
            # Higher stability scores have priority in the sort
            path = list(nx.lexicographical_topological_sort(
                G, key=lambda v: -G.nodes[v]['stability']))
            logger.info(f"Stability-aware topological sort result: {path}")
        
        # Assign partitions greedily based on memory constraints
        partition_strategy = self._greedy_partition_assignment(path)
        
        # Calculate execution time of this initial pipeline
        execution_time = self.calculate_path_time(path, partition_strategy)
        logger.info(f"Initial pipeline execution time: {execution_time:.4f} seconds")
        
        end_time = time.time()
        self.optimization_time += (end_time - start_time)
        logger.info(f"Initial pipeline generated in {end_time - start_time:.2f} seconds")
        
        return path, partition_strategy
    
    def _greedy_partition_assignment(self, path):
        """
        Assign partitions to vehicles greedily based on memory constraints.
        
        Args:
            path: List of vehicle IDs representing execution order
            
        Returns:
            Dict mapping vehicle ID to list of assigned partitions
        """
        partition_strategy = {v_id: [] for v_id in self.cluster.vehicles}
        available_partitions = self.model.unit_partitions.copy()
        
        # Sort partitions by capacity (largest first)
        available_partitions.sort(key=lambda p: p.capacity, reverse=True)
        
        # Assign partitions to vehicles following the path order
        for v_id in path:
            vehicle = self.cluster.vehicles[v_id]
            remaining_memory = vehicle.memory
            
            for partition in available_partitions[:]:  # Create a copy for iteration
                if partition.capacity <= remaining_memory:
                    partition_strategy[v_id].append(partition)
                    remaining_memory -= partition.capacity
                    available_partitions.remove(partition)
        
        # If there are remaining partitions, try to assign them to any vehicle with space
        if available_partitions:
            for partition in available_partitions[:]:
                for v_id in path:
                    vehicle = self.cluster.vehicles[v_id]
                    current_usage = sum(p.capacity for p in partition_strategy[v_id])
                    if current_usage + partition.capacity <= vehicle.memory:
                        partition_strategy[v_id].append(partition)
                        available_partitions.remove(partition)
                        break
        
        return partition_strategy
    
    def dqn_pipeline_generation(self, start_vehicle_id):
        """
        Generate pipeline using DQN approach starting from specified vehicle.
        
        Args:
            start_vehicle_id: ID of the vehicle to start the pipeline from
            
        Returns:
            Tuple of (path, partition_strategy)
        """
        logger.info(f"Starting DQN pipeline generation from vehicle {start_vehicle_id}")
        
        # Initialize with empty partitions
        partition_strategy = {v_id: [] for v_id in self.cluster.vehicles}
        path = [start_vehicle_id]
        
        # Get remaining vehicles to add to the path
        remaining_vehicles = [v_id for v_id in self.cluster.vehicles if v_id != start_vehicle_id]
        
        # Sort remaining vehicles by stability (ascending, as we want less stable vehicles later)
        remaining_vehicles.sort(key=lambda v_id: self.stability_scores[v_id])
        
        # Generate state
        state = self.generate_state(partition_strategy, path)
        
        # Keep track of assigned partitions
        assigned_partitions = set()
        
        # Maximum number of steps to prevent infinite loops
        max_steps = len(self.cluster.vehicles) * 10
        step_count = 0
        
        # Use DQN to iteratively build the pipeline
        while remaining_vehicles and len(assigned_partitions) < len(self.model.unit_partitions) and step_count < max_steps:
            step_count += 1
            logger.debug(f"DQN step {step_count}, remaining vehicles: {len(remaining_vehicles)}, "
                       f"assigned partitions: {len(assigned_partitions)}/{len(self.model.unit_partitions)}")
            
            # Get action from DQN with reduced exploration probability for generation
            # Use a lower epsilon for more exploitation during generation
            original_epsilon = self.dqn_model.epsilon
            self.dqn_model.epsilon = min(0.1, original_epsilon)  # Use at most 10% exploration
            action = self.dqn_model.get_action(state)
            self.dqn_model.epsilon = original_epsilon  # Restore original epsilon
            
            # Decode action into vehicle and partition size
            vehicle_idx = action // 5
            size_level = action % 5
            
            # If action is invalid (index out of range), take a random valid action instead
            if vehicle_idx >= len(remaining_vehicles):
                # Choose a random vehicle
                vehicle_idx = np.random.randint(0, len(remaining_vehicles)) if remaining_vehicles else 0
                size_level = np.random.randint(0, 5)  # Random size level
                logger.debug(f"Invalid action, using random action instead: vehicle_idx={vehicle_idx}, size_level={size_level}")
            
            # Map to actual vehicle ID
            if remaining_vehicles:
                next_vehicle_id = remaining_vehicles[vehicle_idx]
                
                # Check if adding this vehicle maintains DAG constraints
                test_path = path + [next_vehicle_id]
                if self.cluster.is_valid_path(test_path):
                    # Add vehicle to path
                    path.append(next_vehicle_id)
                    remaining_vehicles.remove(next_vehicle_id)
                    logger.debug(f"Added vehicle {next_vehicle_id} to path, size level {size_level}")
                    
                    # Select partitions based on size level
                    previous_assigned = len(assigned_partitions)
                    self._assign_partitions_by_size(
                        next_vehicle_id, size_level, partition_strategy, assigned_partitions)
                    logger.debug(f"Assigned {len(assigned_partitions) - previous_assigned} new partitions to {next_vehicle_id}")
                else:
                    logger.debug(f"Cannot add vehicle {next_vehicle_id} due to DAG constraints")
            
            # Update state
            state = self.generate_state(partition_strategy, path)
            
            # If we've added all vehicles but not all partitions, force assignment of remaining partitions
            if len(path) == len(self.cluster.vehicles) and len(assigned_partitions) < len(self.model.unit_partitions):
                unassigned = [p for p in self.model.unit_partitions if id(p) not in assigned_partitions]
                logger.info(f"All vehicles in path, forcing assignment of {len(unassigned)} remaining partitions")
                self._assign_remaining_partitions(unassigned, partition_strategy)
                
                # Update assigned partitions
                assigned_partitions = set()
                for partitions in partition_strategy.values():
                    for p in partitions:
                        assigned_partitions.add(id(p))
                
                # Break the loop as we're done
                break
        
        # If not all vehicles are in path, add remaining ones in valid order
        if remaining_vehicles:
            logger.info(f"Adding {len(remaining_vehicles)} remaining vehicles in valid topological order")
            G = nx.DiGraph()
            for v_id in self.cluster.vehicles:
                G.add_node(v_id)
            
            for source, target in self.cluster.dependencies:
                G.add_edge(source, target)
            
            # Find valid ordering for remaining vehicles
            topo_sort = list(nx.topological_sort(G))
            for v_id in topo_sort:
                if v_id in remaining_vehicles and v_id not in path:
                    path.append(v_id)
                    logger.debug(f"Added remaining vehicle {v_id} to path via topological sort")
        
        # Assign any remaining partitions
        unassigned = [p for p in self.model.unit_partitions if id(p) not in assigned_partitions]
        if unassigned:
            logger.info(f"Finally assigning {len(unassigned)} remaining partitions")
            self._assign_remaining_partitions(unassigned, partition_strategy)
            
            # Update assigned partitions
            assigned_partitions = set()
            for partitions in partition_strategy.values():
                for p in partitions:
                    assigned_partitions.add(id(p))
        
        logger.info(f"DQN pipeline generation complete: path={path}, assigned_partitions={len(assigned_partitions)}/{len(self.model.unit_partitions)}")
        
        return path, partition_strategy
    
    def _assign_partitions_by_size(self, vehicle_id, size_level, partition_strategy, assigned_partitions):
        """
        Assign partitions to a vehicle based on size level.
        
        Args:
            vehicle_id: ID of the vehicle to assign partitions to
            size_level: Level of memory usage (0-4, with 4 being highest)
            partition_strategy: Current partition strategy to update
            assigned_partitions: Set of IDs of already assigned partitions
        """
        vehicle = self.cluster.vehicles[vehicle_id]
        available_partitions = [p for p in self.model.unit_partitions if id(p) not in assigned_partitions]
        
        if not available_partitions:
            return
        
        # Sort partitions by capacity
        available_partitions.sort(key=lambda p: p.capacity)
        
        # Calculate target memory usage based on size level
        memory_fraction = (size_level + 1) / 5  # 0.2, 0.4, 0.6, 0.8, 1.0
        target_memory = vehicle.memory * memory_fraction
        
        # Assign partitions up to target memory
        current_memory = 0
        for partition in available_partitions:
            if current_memory + partition.capacity <= target_memory:
                partition_strategy[vehicle_id].append(partition)
                assigned_partitions.add(id(partition))
                current_memory += partition.capacity
    
    def _assign_remaining_partitions(self, partitions, partition_strategy):
        """
        Assign any remaining partitions to vehicles with available memory.
        
        Args:
            partitions: List of unassigned partitions
            partition_strategy: Current partition strategy to update
        """
        # Sort partitions by size (largest first for greedy assignment)
        partitions.sort(key=lambda p: p.capacity, reverse=True)
        
        for partition in partitions:
            assigned = False
            
            # Try to assign to each vehicle
            for v_id, vehicle in self.cluster.vehicles.items():
                current_usage = sum(p.capacity for p in partition_strategy[v_id])
                if current_usage + partition.capacity <= vehicle.memory:
                    partition_strategy[v_id].append(partition)
                    assigned = True
                    break
            
            # If couldn't assign, make a best effort (might violate memory constraints)
            if not assigned:
                # Find vehicle with most available memory
                best_vehicle_id = min(
                    self.cluster.vehicles.keys(),
                    key=lambda v_id: (sum(p.capacity for p in partition_strategy[v_id]) / 
                                     self.cluster.vehicles[v_id].memory)
                )
                partition_strategy[best_vehicle_id].append(partition)
    
    def optimize(self):
        """
        Find optimal set of pipelines using SWIFT algorithm.
        
        The SWIFT algorithm follows two phases:
        1. Establish an initial pipeline using stability scores
        2. Generate additional pipelines for remaining vehicles using DQN,
           starting from vehicles with lower stability
        
        Returns:
            List of tuples (path, partition_strategy, execution_time)
        """
        logger.info("Starting SWIFT optimization process")
        start_time = time.time()
        
        # Phase 1: Generate initial pipeline using greedy approach
        logger.info("Phase 1: Generating initial stability-based pipeline")
        initial_path, initial_strategy = self.initial_pipeline_greedy()
        initial_time = self.calculate_path_time(initial_path, initial_strategy)
        
        self.pipelines.append((initial_path, initial_strategy, initial_time))
        logger.info(f"Initial pipeline: {initial_path} with execution time {initial_time:.4f}s")
        
        # Extract the first vehicle from initial pipeline
        initial_first_vehicle = initial_path[0]
        logger.info(f"Initial pipeline starts with vehicle {initial_first_vehicle}, " 
                   f"stability score: {self.stability_scores[initial_first_vehicle]:.3f}")
        
        # Phase 2: Generate pipelines for remaining vehicles using DQN
        logger.info("Phase 2: Generating DQN-based pipelines for remaining vehicles")
        remaining_vehicles = [v_id for v_id in self.cluster.vehicles 
                             if v_id != initial_first_vehicle]
        
        # Sort by stability scores (ascending) - process less stable vehicles first
        # This follows the SWIFT algorithm description where vehicles with lower stability
        # are prioritized in the second phase
        remaining_vehicles.sort(key=lambda v_id: self.stability_scores[v_id])
        
        # Safety check - limit the maximum time for Phase 2
        phase2_timeout = 60  # seconds
        phase2_start = time.time()
        
        # Generate pipeline for each remaining vehicle with timeout safety
        for i, start_vehicle_id in enumerate(remaining_vehicles):
            logger.info(f"Generating pipeline {i+2}/{len(self.cluster.vehicles)} "
                      f"starting with vehicle {start_vehicle_id}, "
                      f"stability score: {self.stability_scores[start_vehicle_id]:.3f}")
            
            # Check if we've exceeded the timeout
            if time.time() - phase2_start > phase2_timeout:
                logger.warning(f"Phase 2 timeout reached after {i+1} of {len(remaining_vehicles)} vehicles. Stopping early.")
                break
                
            try:
                # Set a timeout for this specific vehicle's pipeline generation
                path, strategy = self.dqn_pipeline_generation(start_vehicle_id)
                execution_time = self.calculate_path_time(path, strategy)
                self.pipelines.append((path, strategy, execution_time))
                
                logger.info(f"Pipeline {i+2}: {path} with execution time {execution_time:.4f}s")
            except Exception as e:
                logger.error(f"Error generating pipeline for vehicle {start_vehicle_id}: {e}")
                continue
        
        # Sort pipelines by execution time (fastest first)
        if self.pipelines:
            self.pipelines.sort(key=lambda x: x[2])
            logger.info(f"Generated {len(self.pipelines)} pipelines")
            logger.info(f"Best pipeline: {self.pipelines[0][0]} with time {self.pipelines[0][2]:.4f}s")
        else:
            logger.warning("No valid pipelines were generated!")
        
        end_time = time.time()
        self.optimization_time = end_time - start_time
        logger.info(f"SWIFT optimization completed in {self.optimization_time:.2f} seconds")
        
        # Return all generated pipelines
        return self.pipelines
    
    def calculate_path_time(self, path, partition_strategy):
        """
        Calculate execution time for a path with a specific partition strategy.
        
        Args:
            path: List of vehicle IDs representing execution order
            partition_strategy: Dict mapping vehicle ID to list of assigned partitions
            
        Returns:
            Total execution time
        """
        total_time = 0
        
        # Update vehicle assignments
        for v_id, partitions in partition_strategy.items():
            self.cluster.vehicles[v_id].assigned_partitions = partitions
        
        # Calculate computation and communication times
        for i, v_id in enumerate(path):
            vehicle = self.cluster.vehicles[v_id]
            # Add computation time
            total_time += vehicle.compute_time(
                self.batch_size, self.utilization, self.memory_overhead)
            
            # Add communication time for all but the last vehicle
            if i < len(path) - 1:
                total_time += vehicle.communication_time(
                    self.batch_size, self.memory_overhead)
                
        return total_time
    
    def get_best_pipeline(self):
        """
        Get the best pipeline from all generated pipelines.
        
        Returns:
            Tuple of (path, partition_strategy, execution_time)
        """
        if not self.pipelines:
            return None
        
        # Return pipeline with minimum execution time
        return min(self.pipelines, key=lambda x: x[2])
    
    def train_dqn_model(self, episodes=100, save_path=None):
        """
        Train the DQN model for pipeline generation.
        
        Args:
            episodes: Number of training episodes
            save_path: Path to save the trained model
            
        Returns:
            List of episode rewards
        """
        episode_rewards = []
        
        for episode in tqdm(range(episodes), desc=f"Training Episodes on {self.device}"):
            # Reset environment
            partition_strategy = {v_id: [] for v_id in self.cluster.vehicles}
            vehicle_ids = list(self.cluster.vehicles.keys())
            
            # Randomly select a starting vehicle
            start_vehicle_id = np.random.choice(vehicle_ids)
            path = [start_vehicle_id]
            
            # Initialize state
            state = self.generate_state(partition_strategy, path)
            
            # Keep track of assigned partitions
            assigned_partitions = set()
            
            # Track total reward for this episode
            total_reward = 0
            done = False
            
            # Safety counter to prevent infinite loops
            max_steps = len(self.cluster.vehicles) * 10
            step_count = 0
            
            # Simulate episode
            while not done and step_count < max_steps:
                # Increment step counter
                step_count += 1
                
                # Get action from DQN
                action = self.dqn_model.get_action(state)
                
                # Take action
                next_partition_strategy, next_path, reward, done = self._take_action(
                    action, partition_strategy.copy(), path.copy(), assigned_partitions.copy())
                
                # Check if we're making progress (path or partitions changed)
                path_changed = len(next_path) != len(path)
                partitions_changed = False
                
                new_assigned_partitions = set()
                for partitions in next_partition_strategy.values():
                    for p in partitions:
                        new_assigned_partitions.add(id(p))
                
                partitions_changed = len(new_assigned_partitions) != len(assigned_partitions)
                
                # If no progress is made for this step and we're not done yet,
                # force an early termination to prevent deadlock
                if not path_changed and not partitions_changed and not done:
                    # Randomly select a different action to break potential deadlock
                    # This is a simple way to inject some randomness and break out of stuck states
                    if step_count % 3 == 0:  # Apply this every few steps
                        logger.debug(f"Breaking potential deadlock in episode {episode}, step {step_count}")
                        # Force exploration by overriding with a random action
                        action = np.random.randint(0, self.dqn_model.action_size)
                        next_partition_strategy, next_path, reward, done = self._take_action(
                            action, partition_strategy.copy(), path.copy(), assigned_partitions.copy())
                
                # Update assigned partitions
                assigned_partitions = new_assigned_partitions
                
                # Get next state
                next_state = self.generate_state(next_partition_strategy, next_path)
                
                # Train DQN
                self.dqn_model.train(state, action, reward, next_state, done)
                
                # Update state and path
                state = next_state
                path = next_path.copy()
                partition_strategy = next_partition_strategy.copy()
                
                # Add to total reward
                total_reward += reward
                
                # Episode is done if all partitions are assigned or no more valid actions
                if len(assigned_partitions) == len(self.model.unit_partitions):
                    done = True
                    
                # If we have all vehicles in the path but not all partitions assigned,
                # force assignment of remaining partitions to complete the episode
                if len(path) == len(self.cluster.vehicles) and len(assigned_partitions) < len(self.model.unit_partitions):
                    unassigned = [p for p in self.model.unit_partitions if id(p) not in assigned_partitions]
                    self._assign_remaining_partitions(unassigned, partition_strategy)
                    
                    # Update assigned partitions after final assignment
                    new_assigned_partitions = set()
                    for partitions in partition_strategy.values():
                        for p in partitions:
                            new_assigned_partitions.add(id(p))
                            
                    assigned_partitions = new_assigned_partitions
                    if len(assigned_partitions) == len(self.model.unit_partitions):
                        done = True
            
            # If we reached max steps without completing, log a warning
            if step_count >= max_steps and not done:
                logger.warning(f"Episode {episode} reached max steps without completion. "
                              f"Path: {path}, Partitions: {len(assigned_partitions)}/{len(self.model.unit_partitions)}")
            
            # Update target network periodically
            updated = self.dqn_model.update_target_if_needed(10)  # Update every 10 episodes
            if updated:
                logger.debug(f"Updated target network at episode {episode}")
            
            # Log episode results
            episode_rewards.append(total_reward)
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Total Reward = {total_reward}, Steps = {step_count}")
        
        # Save model if requested
        if save_path:
            self.dqn_model.save_model(save_path)
            
        return episode_rewards
    
    def _take_action(self, action, partition_strategy, path, assigned_partitions):
        """
        Execute an action in the environment.
        
        Args:
            action: Action index
            partition_strategy: Current partition strategy
            path: Current execution path
            assigned_partitions: Set of already assigned partition IDs
            
        Returns:
            Tuple of (next_partition_strategy, next_path, reward, done)
        """
        # Decode action into vehicle and partition size
        vehicle_idx = action // 5
        size_level = action % 5
        
        # Get all vehicle IDs not yet in path
        remaining_vehicles = [v_id for v_id in self.cluster.vehicles if v_id not in path]
        
        # Default values if action is invalid
        reward = -1  # Small negative reward for invalid actions
        done = False
        
        # Check if action is valid
        if vehicle_idx < len(remaining_vehicles):
            next_vehicle_id = remaining_vehicles[vehicle_idx]
            
            # Check if adding this vehicle maintains DAG constraints
            test_path = path + [next_vehicle_id]
            if self.cluster.is_valid_path(test_path):
                # Add vehicle to path
                path.append(next_vehicle_id)
                
                # Select partitions based on size level
                self._assign_partitions_by_size(
                    next_vehicle_id, size_level, partition_strategy, assigned_partitions)
                
                # Calculate reward for this action
                reward = self.calculate_reward(path, partition_strategy)
        
        # Check if we've assigned all partitions
        all_assigned = len(assigned_partitions) == len(self.model.unit_partitions)
        
        # Check if all vehicles are in the path
        all_vehicles_in_path = len(path) == len(self.cluster.vehicles)
        
        # Episode is done if all partitions are assigned or all vehicles are in path
        done = all_assigned or all_vehicles_in_path
        
        return partition_strategy, path, reward, done
    
    def save(self, filepath):
        """
        Save the optimizer state to a file.
        
        Args:
            filepath: Path to save the optimizer state
        """
        # Save DQN model
        self.dqn_model.save_model(f"{filepath}_dqn.pt")
        
        # Save pipelines and stability scores
        torch.save({
            'pipelines': self.pipelines,
            'stability_scores': self.stability_scores
        }, f"{filepath}_swift.pt")
    
    def load(self, filepath):
        """
        Load the optimizer state from a file.
        
        Args:
            filepath: Path to load the optimizer state from
        """
        # Load DQN model
        self.dqn_model.load_model(f"{filepath}_dqn.pt")
        
        # Load pipelines and stability scores
        checkpoint = torch.load(f"{filepath}_swift.pt")
        self.pipelines = checkpoint['pipelines']
        self.stability_scores = checkpoint['stability_scores']
        
    def evaluate_pipeline(self, path, partition_strategy):
        """
        Evaluate a pipeline configuration in detail.
        
        Args:
            path: List of vehicle IDs representing execution order
            partition_strategy: Dict mapping vehicle ID to list of assigned partitions
            
        Returns:
            Dict containing detailed evaluation metrics
        """
        # Update vehicle assignments
        for v_id, partitions in partition_strategy.items():
            self.cluster.vehicles[v_id].assigned_partitions = partitions
            
        # Calculate metrics
        vehicle_metrics = {}
        total_compute_time = 0
        total_comm_time = 0
        
        for i, v_id in enumerate(path):
            vehicle = self.cluster.vehicles[v_id]
            
            # Compute time
            compute_time = vehicle.compute_time(
                self.batch_size, self.utilization, self.memory_overhead)
            total_compute_time += compute_time
            
            # Communication time (except for last vehicle)
            comm_time = 0
            if i < len(path) - 1:
                comm_time = vehicle.communication_time(
                    self.batch_size, self.memory_overhead)
                total_comm_time += comm_time
                
            # Memory usage
            memory_usage = sum(p.capacity for p in partition_strategy.get(v_id, []))
            memory_utilization = memory_usage / vehicle.memory if vehicle.memory > 0 else 0
            
            # Store metrics for this vehicle
            vehicle_metrics[v_id] = {
                'compute_time': compute_time,
                'communication_time': comm_time,
                'memory_usage': memory_usage,
                'memory_utilization': memory_utilization,
                'stability_score': self.stability_scores.get(v_id, 0),
                'partitions': [p.name for p in partition_strategy.get(v_id, [])]
            }
            
        # Calculate overall metrics
        total_time = total_compute_time + total_comm_time
        total_memory = sum(sum(p.capacity for p in partitions) 
                          for partitions in partition_strategy.values())
        
        # Return all metrics
        return {
            'path': path,
            'total_time': total_time,
            'total_compute_time': total_compute_time,
            'total_communication_time': total_comm_time,
            'total_memory': total_memory,
            'vehicle_metrics': vehicle_metrics
        }
    
    def visualize_pipelines(self, save_path=None, show=True):
        """
        Visualize the pipelines generated by SWIFT.
        
        Args:
            save_path: Optional path to save the visualization
            show: Whether to display the plot
        """
        import matplotlib.pyplot as plt
        
        if not self.pipelines:
            logger.warning("No pipelines to visualize.")
            return
        
        # Create a figure with multiple subplots
        n_pipelines = len(self.pipelines)
        fig, axs = plt.subplots(n_pipelines, 1, figsize=(12, 4 * n_pipelines), squeeze=False)
        
        for i, (path, strategy, exec_time) in enumerate(self.pipelines):
            # Get the axis for this pipeline
            ax = axs[i, 0]
            
            # Map vehicles to positions
            positions = {v_id: j for j, v_id in enumerate(path)}
            max_pos = len(path) - 1
            
            # Track current time for each vehicle
            current_times = {v_id: 0 for v_id in path}
            
            # Plot execution blocks
            colors = plt.cm.viridis(np.linspace(0, 1, len(self.model.unit_partitions)))
            color_map = {p.name: colors[i % len(colors)] for i, p in enumerate(self.model.unit_partitions)}
            
            # For each vehicle in the path
            for j, v_id in enumerate(path):
                vehicle = self.cluster.vehicles[v_id]
                partitions = strategy.get(v_id, [])
                
                # Computation time
                compute_time = vehicle.compute_time(self.batch_size, self.utilization, self.memory_overhead)
                
                # Plot computation block
                ax.barh(
                    positions[v_id],
                    compute_time,
                    left=current_times[v_id],
                    height=0.5,
                    color='skyblue',
                    edgecolor='black',
                    alpha=0.7
                )
                
                # Add partition labels
                if partitions:
                    partition_names = ", ".join([p.name for p in partitions])
                    ax.text(
                        current_times[v_id] + compute_time / 2,
                        positions[v_id],
                        partition_names,
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='black'
                    )
                
                # Update current time
                current_times[v_id] += compute_time
                
                # Communication time (if not the last vehicle)
                if j < len(path) - 1:
                    comm_time = vehicle.communication_time(self.batch_size, self.memory_overhead)
                    
                    # Plot communication block
                    ax.barh(
                        positions[v_id],
                        comm_time,
                        left=current_times[v_id],
                        height=0.5,
                        color='lightcoral',
                        edgecolor='black',
                        hatch='/',
                        alpha=0.7
                    )
                    
                    # Update current time
                    current_times[v_id] += comm_time
            
            # Set axis labels and title
            ax.set_yticks(list(positions.values()))
            ax.set_yticklabels([f"Vehicle {v_id}\n(Stability: {self.stability_scores[v_id]:.2f})" 
                               for v_id in positions.keys()])
            ax.set_xlabel("Time (seconds)")
            ax.set_title(f"Pipeline {i+1}: Execution Time = {exec_time:.4f}s")
            
            # Add a line showing total execution time
            ax.axvline(x=exec_time, color='red', linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Pipeline visualization saved to {save_path}")
        
        # Show the plot if requested
        if show:
            plt.show()
            
    def compare_with_optimal(self, optimal_pipeline, optimal_execution_time):
        """
        Compare SWIFT results with optimal pipeline found by exhaustive search.
        
        Args:
            optimal_pipeline: The optimal pipeline found by exhaustive search
            optimal_execution_time: The execution time of the optimal pipeline
            
        Returns:
            Dict with comparison metrics
        """
        if not self.pipelines:
            return {"error": "No pipelines generated by SWIFT"}
        
        # Get the best pipeline found by SWIFT
        swift_best = min(self.pipelines, key=lambda x: x[2])
        swift_path, swift_strategy, swift_time = swift_best
        
        # Calculate the optimality gap
        gap_absolute = swift_time - optimal_execution_time
        gap_percentage = (gap_absolute / optimal_execution_time) * 100
        
        # Find the rank of the optimal pipeline in SWIFT results
        optimal_rank = None
        for i, (path, strategy, time) in enumerate(sorted(self.pipelines, key=lambda x: x[2])):
            if path == optimal_pipeline:
                optimal_rank = i + 1
                break
        
        # Calculate the speedup in optimization time (assuming exhaustive search took longer)
        # This would need the actual exhaustive search time in a real comparison
        
        return {
            "swift_best_path": swift_path,
            "swift_best_time": swift_time,
            "optimal_path": optimal_pipeline,
            "optimal_time": optimal_execution_time,
            "absolute_gap": gap_absolute,
            "percentage_gap": gap_percentage,
            "optimal_rank_in_swift": optimal_rank,
            "optimization_time": self.optimization_time,
        }