import xml.etree.ElementTree as ET
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from vehicle import Vehicle, FLADCluster

def load_cluster_from_xml(xml_path):
    """
    Load a FLAD cluster configuration from an XML file.
    
    The XML format should be:
    
    <cluster name="ClusterName">
        <vehicles>
            <vehicle id="v1" memory="4e9" comp_capability="8e9" comm_capability="1e9" />
            <vehicle id="v2" memory="5e9" comp_capability="12e9" comm_capability="1.2e9" />
            <!-- more vehicles... -->
        </vehicles>
        <dependencies>
            <dependency source="v1" target="v3" />
            <dependency source="v2" target="v4" />
            <!-- more dependencies... -->
        </dependencies>
    </cluster>
    
    Args:
        xml_path: Path to the XML file containing cluster configuration
        
    Returns:
        FLADCluster object containing the vehicles and dependencies
    """
    # Check if the file exists
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML format: {e}")
    
    # Get cluster name
    cluster_name = root.get('name', 'UnnamedCluster')
    cluster = FLADCluster(cluster_name)
    
    # Parse vehicles
    vehicles_elem = root.find('vehicles')
    if vehicles_elem is None:
        raise ValueError("No vehicles section found in XML")
    
    for vehicle_elem in vehicles_elem.findall('vehicle'):
        # Get vehicle attributes
        v_id = vehicle_elem.get('id')
        if v_id is None:
            raise ValueError("Vehicle missing required 'id' attribute")
        
        try:
            # Convert values from scientific notation strings to float
            memory = float(eval(vehicle_elem.get('memory', '0')))
            comp_capability = float(eval(vehicle_elem.get('comp_capability', '0')))
            comm_capability = float(eval(vehicle_elem.get('comm_capability', '0')))
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid numeric value for vehicle {v_id}: {e}")
        
        # Create and add vehicle to cluster
        vehicle = Vehicle(v_id, memory, comp_capability, comm_capability)
        cluster.add_vehicle(vehicle)
    
    # Parse dependencies
    dependencies_elem = root.find('dependencies')
    if dependencies_elem is not None:
        for dep_elem in dependencies_elem.findall('dependency'):
            source = dep_elem.get('source')
            target = dep_elem.get('target')
            
            if source is None or target is None:
                raise ValueError("Dependency missing required 'source' or 'target' attribute")
            
            try:
                cluster.add_dependency(source, target)
            except ValueError as e:
                raise ValueError(f"Invalid dependency {source} -> {target}: {e}")
    
    return cluster

def generate_cluster_dag(cluster, output_path=None, show=True):
    """
    Generate a directed acyclic graph visualization of the cluster dependencies.
    
    Args:
        cluster: FLADCluster object
        output_path: Optional path to save the visualization
        show: Whether to display the visualization
        
    Returns:
        networkx.DiGraph object representing the cluster DAG
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (vehicles)
    for v_id, vehicle in cluster.vehicles.items():
        G.add_node(v_id, 
                  memory=vehicle.memory, 
                  comp_capability=vehicle.comp_capability,
                  comm_capability=vehicle.comm_capability)
    
    # Add edges (dependencies)
    for source, target in cluster.dependencies:
        G.add_edge(source, target)
    
    # Check if it's a DAG
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"The graph contains cycles: {cycles}")
    
    # Visualize the DAG if requested
    if output_path is not None or show:
        plt.figure(figsize=(10, 8))
        
        # Try to use hierarchical layout if pygraphviz is available
        # Otherwise fall back to spring layout with a fixed seed for reproducibility
        try:
            # Check if pygraphviz is available
            try:
                import pygraphviz
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except ImportError:
                print("Warning: pygraphviz not found, using spring layout instead.")
                pos = nx.spring_layout(G, seed=42)
        except Exception as e:
            print(f"Graph layout error: {e}. Using spring layout instead.")
            pos = nx.spring_layout(G, seed=42)
        
        # Use a maximum value for normalization to prevent division by zero
        max_memory = max([v.memory for v in cluster.vehicles.values()] + [1e-9])
        max_comp = max([v.comp_capability for v in cluster.vehicles.values()] + [1e-9])
        
        # Node sizes based on memory
        node_sizes = [1000 * cluster.vehicles[v_id].memory / max_memory 
                      for v_id in G.nodes()]
        
        # Node colors based on computational capability
        node_colors = [cluster.vehicles[v_id].comp_capability / max_comp 
                      for v_id in G.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.8, cmap='viridis')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Add labels with vehicle details
        vehicle_labels = {v_id: f"{v_id}\nMem: {cluster.vehicles[v_id].memory/1e9:.1f}GB\n"
                               f"Comp: {cluster.vehicles[v_id].comp_capability/1e9:.1f}GFLOPS\n"
                               f"Comm: {cluster.vehicles[v_id].comm_capability/1e9:.1f}GB/s" 
                         for v_id in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=vehicle_labels, font_size=8)
        
        plt.title(f'Cluster DAG: {cluster.name}')
        plt.axis('off')
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"DAG visualization saved to: {output_path}")
        
        # Show if requested
        if show:
            plt.show()
    
    return G

def save_cluster_to_xml(cluster, xml_path):
    """
    Save a FLAD cluster configuration to an XML file.
    
    Args:
        cluster: FLADCluster object
        xml_path: Path to save the XML file
    """
    # Create root element
    root = ET.Element('cluster')
    root.set('name', cluster.name)
    
    # Add vehicles section
    vehicles_elem = ET.SubElement(root, 'vehicles')
    for v_id, vehicle in cluster.vehicles.items():
        vehicle_elem = ET.SubElement(vehicles_elem, 'vehicle')
        vehicle_elem.set('id', v_id)
        vehicle_elem.set('memory', str(vehicle.memory))
        vehicle_elem.set('comp_capability', str(vehicle.comp_capability))
        vehicle_elem.set('comm_capability', str(vehicle.comm_capability))
    
    # Add dependencies section
    if cluster.dependencies:
        dependencies_elem = ET.SubElement(root, 'dependencies')
        for source, target in cluster.dependencies:
            dep_elem = ET.SubElement(dependencies_elem, 'dependency')
            dep_elem.set('source', source)
            dep_elem.set('target', target)
    
    # Create XML tree and write to file
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"Cluster configuration saved to: {xml_path}")

def generate_random_cluster(name, num_vehicles=5, edge_probability=0.3, 
                          min_memory=2e9, max_memory=8e9, 
                          min_comp=5e9, max_comp=15e9, 
                          min_comm=0.5e9, max_comm=2e9, 
                          seed=None):
    """
    Generate a random cluster with vehicles and dependencies.
    
    Args:
        name: Cluster name
        num_vehicles: Number of vehicles to generate
        edge_probability: Probability of adding an edge between any two vehicles
        min_memory, max_memory: Range for memory allocation
        min_comp, max_comp: Range for computation capability
        min_comm, max_comm: Range for communication capability
        seed: Random seed for reproducibility
        
    Returns:
        FLADCluster object with randomly generated vehicles and dependencies
    """
    # Set random seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
    
    # Create cluster
    cluster = FLADCluster(name)
    
    # Create vehicles
    for i in range(num_vehicles):
        v_id = f"v{i+1}"
        memory = np.random.uniform(min_memory, max_memory)
        comp_capability = np.random.uniform(min_comp, max_comp)
        comm_capability = np.random.uniform(min_comm, max_comm)
        
        vehicle = Vehicle(v_id, memory, comp_capability, comm_capability)
        cluster.add_vehicle(vehicle)
    
    # Generate a random DAG
    # To ensure it's a DAG, we'll randomly add edges only from lower to higher indices
    vehicle_ids = list(cluster.vehicles.keys())
    for i in range(len(vehicle_ids)):
        for j in range(i+1, len(vehicle_ids)):
            if np.random.random() < edge_probability:
                # Add edge from lower to higher index to ensure DAG property
                cluster.add_dependency(vehicle_ids[i], vehicle_ids[j])
    
    return cluster

# Example usage
if __name__ == "__main__":
    # Generate a random cluster
    random_cluster = generate_random_cluster("RandomCluster", num_vehicles=8, seed=42)
    
    # Save to XML
    save_cluster_to_xml(random_cluster, "random_cluster.xml")
    
    # Load from XML
    loaded_cluster = load_cluster_from_xml("random_cluster.xml")
    
    # Visualize the DAG
    generate_cluster_dag(loaded_cluster, output_path="random_cluster_dag.png")
    
    print(f"Generated cluster with {len(loaded_cluster.vehicles)} vehicles and "
          f"{len(loaded_cluster.dependencies)} dependencies")
