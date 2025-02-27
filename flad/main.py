import numpy as np
from model import ModelComponent, UnitModelPartition, DNNModel
from vehicle import Vehicle, FLADCluster
from optimizer import PipelineOptimizer
from visualizer import plot_cluster_dag, plot_pipeline_schedule

def create_sample_model():
    """Create a sample DNN model with components and partitions."""
    # Create model components
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
    
    return model

def create_sample_cluster():
    """Create a sample FLAD cluster with vehicles."""
    # Create vehicles with different capabilities
    v1 = Vehicle("v1", memory=4e9, comp_capability=8e9, comm_capability=1e9)
    v2 = Vehicle("v2", memory=5e9, comp_capability=12e9, comm_capability=1.2e9)
    v3 = Vehicle("v3", memory=3e9, comp_capability=6e9, comm_capability=0.8e9)
    v4 = Vehicle("v4", memory=6e9, comp_capability=10e9, comm_capability=1.5e9)
    v5 = Vehicle("v5", memory=7e9, comp_capability=14e9, comm_capability=1.8e9)
    
    # Create cluster
    cluster = FLADCluster("Sample_Cluster")
    cluster.add_vehicle(v1)
    cluster.add_vehicle(v2)
    cluster.add_vehicle(v3)
    cluster.add_vehicle(v4)
    cluster.add_vehicle(v5)
    
    # Add some dependencies
    cluster.add_dependency("v1", "v3")  # v1 must complete before v3 starts
    cluster.add_dependency("v2", "v4")  # v2 must complete before v4 starts
    cluster.add_dependency("v3", "v4")  # v3 must complete before v4 starts
    cluster.add_dependency("v1", "v5")  # v1 must complete before v5 starts
    cluster.add_dependency("v2", "v5")  # v2 must complete before v5 starts
    
    return cluster

def main():
    """Main function to demonstrate pipeline optimization."""
    # Create sample model and cluster
    model = create_sample_model()
    cluster = create_sample_cluster()
    
    print(f"Created model: {model}")
    print(f"Created cluster with vehicles: {list(cluster.vehicles.values())}")
    
    # Visualize the cluster DAG
    plot_cluster_dag(cluster)
    
    # Create optimizer
    optimizer = PipelineOptimizer(
        cluster, 
        model, 
        batch_size=32,
        utilization=0.5, 
        memory_overhead=1.2
    )
    
    # Find optimal solution
    print("Finding optimal pipeline configuration...")
    optimal_path, optimal_strategy, min_time = optimizer.optimize()
    
    print(f"\nOptimal execution path: {optimal_path}")
    print(f"Minimum execution time: {min_time:.4f} seconds")
    
    print("\nOptimal partition strategy:")
    for v_id, partitions in optimal_strategy.items():
        partition_names = [p.name for p in partitions]
        print(f"  Vehicle {v_id}: {partition_names}")
    
    # Calculate execution times for visualization
    execution_times = {}
    for v_id in cluster.vehicles:
        vehicle = cluster.vehicles[v_id]
        execution_times[(v_id, 'computation')] = vehicle.compute_time(
            optimizer.batch_size, optimizer.utilization, optimizer.memory_overhead)
        execution_times[(v_id, 'communication')] = vehicle.communication_time(
            optimizer.batch_size, optimizer.memory_overhead)
    
    # Visualize the pipeline schedule
    plot_pipeline_schedule(optimal_path, optimal_strategy, execution_times)

if __name__ == "__main__":
    main()
