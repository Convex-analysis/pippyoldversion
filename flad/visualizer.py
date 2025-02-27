import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_cluster_dag(cluster, save_path=None, figsize=(10, 8)):
    """
    Plot the dependency graph of vehicles in a cluster.
    
    Args:
        cluster: FLADCluster object
        save_path: Path to save the figure (optional)
        figsize: Figure size
    """
    # Get adjacency matrix
    adj_matrix, vehicle_ids = cluster.get_adjacency_matrix()
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, v_id in enumerate(vehicle_ids):
        vehicle = cluster.vehicles[v_id]
        G.add_node(v_id, memory=vehicle.memory, 
                  comp_capability=vehicle.comp_capability,
                  comm_capability=vehicle.comm_capability)
    
    # Add edges
    for i, source_id in enumerate(vehicle_ids):
        for j, target_id in enumerate(vehicle_ids):
            if adj_matrix[i, j] == 1:
                G.add_edge(source_id, target_id)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Use spring layout to position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_sizes = [cluster.vehicles[v_id].memory / 1e8 for v_id in G.nodes()]
    node_colors = [cluster.vehicles[v_id].comp_capability / 1e9 for v_id in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color=node_colors, alpha=0.8, cmap='viridis')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    
    # Draw labels
    labels = {v_id: f"Vehicle {v_id}\nMem: {cluster.vehicles[v_id].memory/1e9:.1f}GB\nComp: {cluster.vehicles[v_id].comp_capability/1e9:.1f}GFLOPS\nComm: {cluster.vehicles[v_id].comm_capability/1e9:.1f}GB/s" 
              for v_id in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title(f'Cluster Dependency Graph: {cluster.name}')
    plt.axis('off')
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Cluster DAG visualization saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_pipeline_schedule(path, partition_strategy, execution_times, save_path=None, figsize=(12, 6)):
    """
    Plot the execution schedule of a pipeline.
    
    Args:
        path: List of vehicle IDs in execution order
        partition_strategy: Dict mapping vehicle ID to list of partitions
        execution_times: Dict mapping (vehicle_id, type) to time, where type is 'computation' or 'communication'
        save_path: Path to save the figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Map vehicles to positions
    positions = {v_id: len(path) - 1 - i for i, v_id in enumerate(path)}
    
    # Track current time for each vehicle
    current_times = {v_id: 0 for v_id in path}
    
    # Plot execution blocks
    for v_id in path:
        # Computation block
        computation_time = execution_times.get((v_id, 'computation'), 0)
        plt.barh(positions[v_id], computation_time, left=current_times[v_id], 
                height=0.5, color='skyblue', alpha=0.7, label='Computation')
        
        # Add partition labels
        partitions = partition_strategy.get(v_id, [])
        if partitions:
            partition_names = ", ".join([p.name for p in partitions])
            plt.text(current_times[v_id] + computation_time/2, positions[v_id], 
                    partition_names, ha='center', va='center', fontsize=8)
        
        # Update current time
        current_times[v_id] += computation_time
        
        # Communication block
        communication_time = execution_times.get((v_id, 'communication'), 0)
        if communication_time > 0:
            plt.barh(positions[v_id], communication_time, left=current_times[v_id], 
                    height=0.5, color='lightcoral', alpha=0.7, 
                    hatch='//', label='Communication')
            current_times[v_id] += communication_time
    
    # Set labels and title
    plt.yticks(list(positions.values()), [f"Vehicle {v_id}" for v_id in positions.keys()])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Vehicle')
    plt.title('Pipeline Execution Schedule')
    
    # Add legend (only once for each type)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    # Set grid
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Calculate total execution time
    total_time = max(current_times.values())
    plt.axvline(x=total_time, color='red', linestyle='--')
    plt.text(total_time, len(path), f"Total: {total_time:.4f}s", 
             ha='right', va='bottom', color='red')
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Pipeline schedule visualization saved to {save_path}")
    
    plt.tight_layout()
    plt.show()
