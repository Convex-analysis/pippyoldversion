import os
import matplotlib.pyplot as plt
from vehicle import Vehicle, FLADCluster
from cluster_util import (
    load_cluster_from_xml, 
    save_cluster_to_xml, 
    generate_cluster_dag,
    generate_random_cluster
)
from model import ModelComponent, UnitModelPartition, DNNModel
from swift_optimizer import SWIFTOptimizer

def create_manual_cluster():
    """Create a cluster manually by adding vehicles and dependencies."""
    print("\n1. Creating cluster manually...")
    
    # Create a new cluster
    cluster = FLADCluster("ManualCluster")
    
    # Add vehicles with different capabilities
    cluster.add_vehicle(Vehicle("v1", memory=4e9, comp_capability=8e9, comm_capability=1e9))
    cluster.add_vehicle(Vehicle("v2", memory=5e9, comp_capability=12e9, comm_capability=1.2e9))
    cluster.add_vehicle(Vehicle("v3", memory=3e9, comp_capability=6e9, comm_capability=0.8e9))
    cluster.add_vehicle(Vehicle("v4", memory=6e9, comp_capability=10e9, comm_capability=1.5e9))
    
    # Add dependencies between vehicles
    cluster.add_dependency("v1", "v3")  # v1 must complete before v3
    cluster.add_dependency("v2", "v4")  # v2 must complete before v4
    
    # Visualize the cluster
    generate_cluster_dag(cluster, output_path="manual_cluster.png", show=False)
    print(f"  Created cluster with {len(cluster.vehicles)} vehicles and {len(cluster.dependencies)} dependencies")
    print("  Visualization saved to manual_cluster.png")
    
    return cluster

def create_from_xml():
    """Create a cluster by loading from an XML file."""
    print("\n2. Creating cluster from XML...")
    
    # Define XML content as a string for demonstration
    xml_content = """<?xml version='1.0' encoding='utf-8'?>
    <cluster name="XMLCluster">
        <vehicles>
            <vehicle id="v1" memory="4e9" comp_capability="8e9" comm_capability="1e9" />
            <vehicle id="v2" memory="5e9" comp_capability="12e9" comm_capability="1.2e9" />
            <vehicle id="v3" memory="3e9" comp_capability="6e9" comm_capability="0.8e9" />
            <vehicle id="v4" memory="6e9" comp_capability="10e9" comm_capability="1.5e9" />
            <vehicle id="v5" memory="7e9" comp_capability="15e9" comm_capability="1.8e9" />
        </vehicles>
        <dependencies>
            <dependency source="v1" target="v3" />
            <dependency source="v2" target="v4" />
            <dependency source="v3" target="v5" />
            <dependency source="v4" target="v5" />
        </dependencies>
    </cluster>
    """
    
    # Write the XML content to a file
    xml_path = "temp_cluster.xml"
    with open(xml_path, "w") as f:
        f.write(xml_content)
    
    # Load cluster from the XML file
    cluster = load_cluster_from_xml(xml_path)
    
    # Visualize the cluster
    generate_cluster_dag(cluster, output_path="xml_cluster.png", show=False)
    print(f"  Loaded cluster with {len(cluster.vehicles)} vehicles and {len(cluster.dependencies)} dependencies")
    print("  Visualization saved to xml_cluster.png")
    
    # Clean up the temporary file
    os.remove(xml_path)
    
    return cluster

def create_random_cluster():
    """Create a random cluster using the utility function."""
    print("\n3. Creating random cluster...")
    
    # Generate a random cluster with 8 vehicles
    cluster = generate_random_cluster(
        name="RandomCluster", 
        num_vehicles=8,
        edge_probability=0.3,  # 30% chance of dependency between any two vehicles
        min_memory=3e9, 
        max_memory=8e9,
        min_comp=5e9, 
        max_comp=15e9,
        min_comm=0.5e9, 
        max_comm=2e9,
        seed=42  # For reproducible results
    )
    
    # Save the cluster configuration to XML
    save_cluster_to_xml(cluster, "random_cluster.xml")
    
    # Visualize the cluster
    generate_cluster_dag(cluster, output_path="random_cluster.png", show=False)
    print(f"  Generated cluster with {len(cluster.vehicles)} vehicles and {len(cluster.dependencies)} dependencies")
    print("  Cluster configuration saved to random_cluster.xml")
    print("  Visualization saved to random_cluster.png")
    
    return cluster

def modify_and_save_cluster(cluster):
    """Modify an existing cluster and save it back to XML."""
    print("\n4. Modifying and saving cluster...")
    
    # Add a new vehicle
    cluster.add_vehicle(Vehicle("new_v", memory=10e9, comp_capability=20e9, comm_capability=2.5e9))
    
    # Add new dependencies
    if "v1" in cluster.vehicles and "new_v" in cluster.vehicles:
        cluster.add_dependency("v1", "new_v")
    
    # Save the modified cluster
    save_cluster_to_xml(cluster, "modified_cluster.xml")
    
    # Visualize the modified cluster
    generate_cluster_dag(cluster, output_path="modified_cluster.png", show=False)
    print(f"  Modified cluster now has {len(cluster.vehicles)} vehicles and {len(cluster.dependencies)} dependencies")
    print("  Modified cluster saved to modified_cluster.xml")
    print("  Visualization saved to modified_cluster.png")
    
    return cluster

def use_cluster_with_optimizer(cluster):
    """Use the cluster with a SWIFT optimizer."""
    print("\n5. Using cluster with SWIFT optimizer...")
    
    # Create a simple model
    model = DNNModel("SimpleModel")
    
    # Add components to the model
    c1 = ModelComponent("Component1", flops_per_sample=5e9, capacity=2e9)
    c2 = ModelComponent("Component2", flops_per_sample=8e9, capacity=3e9)
    c3 = ModelComponent("Component3", flops_per_sample=3e9, capacity=1.5e9)
    model.add_component(c1)
    model.add_component(c2)
    model.add_component(c3)
    
    # Add unit partitions
    p1 = UnitModelPartition("Partition1", [c1], c1.flops_per_sample, c1.capacity, communication_volume=0.5e9)
    p2 = UnitModelPartition("Partition2", [c2], c2.flops_per_sample, c2.capacity, communication_volume=0.8e9)
    p3 = UnitModelPartition("Partition3", [c3], c3.flops_per_sample, c3.capacity, communication_volume=0.3e9)
    model.add_unit_partition(p1)
    model.add_unit_partition(p2)
    model.add_unit_partition(p3)
    
    # Create optimizer with the loaded cluster and model
    optimizer = SWIFTOptimizer(
        cluster, 
        model, 
        batch_size=32,
        utilization=0.5,
        memory_overhead=1.2
    )
    
    # Train the optimizer (minimal training just to demonstrate)
    rewards = optimizer.train_dqn_model(episodes=10)
    
    # Get stability scores
    print("  Vehicle stability scores:")
    for v_id, score in optimizer.stability_scores.items():
        print(f"    {v_id}: {score:.4f}")
    
    # Run optimization
    print("  Running optimization...")
    pipelines = optimizer.optimize()
    
    if pipelines:
        best_path, best_strat, best_time = pipelines[0]
        print(f"  Best pipeline: {best_path}")
        print(f"  Best execution time: {best_time:.4f}s")
    else:
        print("  No valid pipelines found.")
    
    return optimizer

def main():
    print("Cluster Utility Example".center(80, "="))
    
    # 1. Create a cluster manually
    manual_cluster = create_manual_cluster()
    
    # 2. Create a cluster from an XML file
    xml_cluster = create_from_xml()
    
    # 3. Create a random cluster
    random_cluster = create_random_cluster()
    
    # 4. Modify and save a cluster
    modified_cluster = modify_and_save_cluster(random_cluster.copy())
    
    # 5. Use the cluster with an optimizer
    optimizer = use_cluster_with_optimizer(xml_cluster)
    
    print("\nAll cluster operations completed successfully!".center(80, "="))

if __name__ == "__main__":
    #main()
    cluster = load_cluster_from_xml("flad\settings\clusterwith3member.xml")
    
    # Visualize the cluster
    generate_cluster_dag(cluster, output_path="./xml_cluster.png", show=False)
    print(f"  Loaded cluster with {len(cluster.vehicles)} vehicles and {len(cluster.dependencies)} dependencies")
    print("  Visualization saved to xml_cluster.png")
