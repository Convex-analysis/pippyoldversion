import os
from model import ModelComponent, UnitModelPartition, DNNModel
from model_util import (
    load_model_from_xml, 
    save_model_to_xml, 
    generate_random_model,
    model_to_graphviz
)
from swift_optimizer import SWIFTOptimizer
from cluster_util import load_cluster_from_xml, generate_random_cluster
import matplotlib.pyplot as plt
import numpy as np

def create_manual_model():
    """Create a DNN model manually with components and partitions."""
    print("\n1. Creating model manually...")
    
    # Create model components
    rgb_backbone = ModelComponent("RGB_Backbone", flops_per_sample=5e9, capacity=2e9)
    lidar_backbone = ModelComponent("Lidar_Backbone", flops_per_sample=8e9, capacity=3e9)
    encoder = ModelComponent("Encoder", flops_per_sample=3e9, capacity=1.5e9)
    decoder = ModelComponent("BEV_Decoder", flops_per_sample=4e9, capacity=2e9)
    
    # Create model and add components
    model = DNNModel("BEVFusion")
    model.add_component(rgb_backbone)
    model.add_component(lidar_backbone)
    model.add_component(encoder)
    model.add_component(decoder)
    
    # Create unit partitions
    partition1 = UnitModelPartition(
        "RGB_Only", 
        [rgb_backbone], 
        rgb_backbone.flops_per_sample, 
        rgb_backbone.capacity, 
        communication_volume=0.5e9
    )
    
    partition2 = UnitModelPartition(
        "Lidar_Only", 
        [lidar_backbone], 
        lidar_backbone.flops_per_sample, 
        lidar_backbone.capacity, 
        communication_volume=0.8e9
    )
    
    partition3 = UnitModelPartition(
        "Encoder_Decoder", 
        [encoder, decoder], 
        encoder.flops_per_sample + decoder.flops_per_sample, 
        encoder.capacity + decoder.capacity, 
        communication_volume=1.2e9
    )
    
    # Add partitions to model
    model.add_unit_partition(partition1)
    model.add_unit_partition(partition2)
    model.add_unit_partition(partition3)
    
    # Create visualization
    model_to_graphviz(model, "manual_model")
    
    print(f"  Created model '{model.name}' with {len(model.components)} components and {len(model.unit_partitions)} partitions")
    print(f"  Total capacity: {model.total_capacity / 1e9:.2f} GB")
    print(f"  Total FLOPs per sample: {model.total_flops_per_sample / 1e9:.2f} GFLOPs")
    print("  Visualization saved to manual_model.png")
    
    return model

def create_from_xml():
    """Create a model by loading from an XML file."""
    print("\n2. Creating model from XML...")
    
    # Define XML content as a string for demonstration
    xml_content = """<?xml version='1.0' encoding='utf-8'?>
    <model name="XMLModel">
        <components>
            <component name="InputLayer" flops_per_sample="2e9" capacity="1e9" />
            <component name="ConvBlock1" flops_per_sample="6e9" capacity="2.5e9" />
            <component name="ConvBlock2" flops_per_sample="8e9" capacity="3e9" />
            <component name="ConvBlock3" flops_per_sample="10e9" capacity="4e9" />
            <component name="OutputLayer" flops_per_sample="1e9" capacity="0.5e9" />
        </components>
        <partitions>
            <partition name="Frontend" communication_volume="0.7e9">
                <component_ref name="InputLayer" />
                <component_ref name="ConvBlock1" />
            </partition>
            <partition name="Backend" communication_volume="1.2e9">
                <component_ref name="ConvBlock2" />
                <component_ref name="ConvBlock3" />
            </partition>
            <partition name="Output" communication_volume="0.3e9">
                <component_ref name="OutputLayer" />
            </partition>
        </partitions>
    </model>
    """
    
    # Write the XML content to a file
    xml_path = "temp_model.xml"
    with open(xml_path, "w") as f:
        f.write(xml_content)
    
    # Load model from the XML file
    model = load_model_from_xml(xml_path)
    
    # Create visualization
    model_to_graphviz(model, "xml_model")
    
    print(f"  Loaded model '{model.name}' with {len(model.components)} components and {len(model.unit_partitions)} partitions")
    print(f"  Total capacity: {model.total_capacity / 1e9:.2f} GB")
    print(f"  Total FLOPs per sample: {model.total_flops_per_sample / 1e9:.2f} GFLOPs")
    print("  Visualization saved to xml_model.png")
    
    # Clean up the temporary file
    os.remove(xml_path)
    
    return model

def create_random_model():
    """Create a random model using the utility function."""
    print("\n3. Creating random model...")
    
    # Generate a random model
    model = generate_random_model(
        name="RandomModel",
        num_components=8,
        num_partitions=5,
        min_flops=1e9,
        max_flops=10e9,
        min_capacity=0.5e9,
        max_capacity=4e9,
        min_comm=0.2e9,
        max_comm=2e9,
        seed=42  # For reproducible results
    )
    
    # Save the model configuration to XML
    save_model_to_xml(model, "random_model.xml")
    
    # Create visualization
    model_to_graphviz(model, "random_model")
    
    print(f"  Generated model '{model.name}' with {len(model.components)} components and {len(model.unit_partitions)} partitions")
    print(f"  Total capacity: {model.total_capacity / 1e9:.2f} GB")
    print(f"  Total FLOPs per sample: {model.total_flops_per_sample / 1e9:.2f} GFLOPs")
    print("  Model configuration saved to random_model.xml")
    print("  Visualization saved to random_model.png")
    
    return model

def modify_and_save_model(model):
    """Modify an existing model and save it back to XML."""
    print("\n4. Modifying and saving model...")
    
    # Create a new component
    new_component = ModelComponent("New_Component", flops_per_sample=7e9, capacity=2.8e9)
    model.add_component(new_component)
    
    # Create a new partition with the new component
    new_partition = UnitModelPartition(
        "New_Partition", 
        [new_component], 
        new_component.flops_per_sample, 
        new_component.capacity, 
        communication_volume=0.9e9
    )
    model.add_unit_partition(new_partition)
    
    # Save the modified model
    save_model_to_xml(model, "modified_model.xml")
    
    # Create visualization
    model_to_graphviz(model, "modified_model")
    
    print(f"  Modified model now has {len(model.components)} components and {len(model.unit_partitions)} partitions")
    print(f"  Total capacity: {model.total_capacity / 1e9:.2f} GB")
    print(f"  Total FLOPs per sample: {model.total_flops_per_sample / 1e9:.2f} GFLOPs")
    print("  Model saved to modified_model.xml")
    print("  Visualization saved to modified_model.png")
    
    return model

def analyze_model_and_partitions(model):
    """Analyze a model's components and partitions."""
    print("\n5. Analyzing model and partitions...")
    
    # Calculate metrics
    flops_per_component = [c.flops_per_sample / 1e9 for c in model.components]
    capacity_per_component = [c.capacity / 1e9 for c in model.components]
    flops_per_partition = [p.flops_per_sample / 1e9 for p in model.unit_partitions]
    capacity_per_partition = [p.capacity / 1e9 for p in model.unit_partitions]
    comm_per_partition = [p.communication_volume / 1e9 for p in model.unit_partitions]
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot component FLOPS
    axs[0, 0].bar(range(len(model.components)), flops_per_component, color='skyblue')
    axs[0, 0].set_title('Component FLOPS')
    axs[0, 0].set_xlabel('Component Index')
    axs[0, 0].set_ylabel('GFLOPS per Sample')
    axs[0, 0].set_xticks(range(len(model.components)))
    axs[0, 0].set_xticklabels([c.name for c in model.components], rotation=45, ha='right')
    
    # Plot component capacity
    axs[0, 1].bar(range(len(model.components)), capacity_per_component, color='lightgreen')
    axs[0, 1].set_title('Component Capacity')
    axs[0, 1].set_xlabel('Component Index')
    axs[0, 1].set_ylabel('Capacity (GB)')
    axs[0, 1].set_xticks(range(len(model.components)))
    axs[0, 1].set_xticklabels([c.name for c in model.components], rotation=45, ha='right')
    
    # Plot partition FLOPS
    axs[1, 0].bar(range(len(model.unit_partitions)), flops_per_partition, color='coral')
    axs[1, 0].set_title('Partition FLOPS')
    axs[1, 0].set_xlabel('Partition Index')
    axs[1, 0].set_ylabel('GFLOPS per Sample')
    axs[1, 0].set_xticks(range(len(model.unit_partitions)))
    axs[1, 0].set_xticklabels([p.name for p in model.unit_partitions], rotation=45, ha='right')
    
    # Plot partition metrics
    x = range(len(model.unit_partitions))
    width = 0.35
    axs[1, 1].bar(np.array(x) - width/2, capacity_per_partition, width, color='lightgreen', label='Capacity (GB)')
    axs[1, 1].bar(np.array(x) + width/2, comm_per_partition, width, color='orchid', label='Communication (GB)')
    axs[1, 1].set_title('Partition Memory Metrics')
    axs[1, 1].set_xlabel('Partition Index')
    axs[1, 1].set_ylabel('GB')
    axs[1, 1].legend()
    axs[1, 1].set_xticks(range(len(model.unit_partitions)))
    axs[1, 1].set_xticklabels([p.name for p in model.unit_partitions], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("model_analysis.png")
    plt.show()
    
    print("  Model analysis complete and saved to model_analysis.png")
    
    return {
        'flops_per_component': flops_per_component,
        'capacity_per_component': capacity_per_component,
        'flops_per_partition': flops_per_partition,
        'capacity_per_partition': capacity_per_partition,
        'comm_per_partition': comm_per_partition
    }

def use_model_with_optimizer(model):
    """Use the model with a SWIFT optimizer."""
    print("\n6. Using model with SWIFT optimizer...")
    
    # Create or load a cluster
    try:
        # Try to load an existing cluster from XML
        cluster = load_cluster_from_xml("random_cluster.xml")
        print("  Loaded cluster from random_cluster.xml")
    except (FileNotFoundError, ValueError):
        # If loading fails, generate a random cluster
        cluster = generate_random_cluster(
            "TestCluster", 
            num_vehicles=5, 
            edge_probability=0.3,
            seed=42
        )
        print("  Generated a random cluster")
    
    # Create optimizer with the model and cluster
    optimizer = SWIFTOptimizer(
        cluster, 
        model, 
        batch_size=32,
        utilization=0.5,
        memory_overhead=1.2
    )
    
    # Train the optimizer (minimal training just to demonstrate)
    print("  Training DQN model with minimal episodes...")
    rewards = optimizer.train_dqn_model(episodes=5)
    
    # Run optimization
    print("  Running optimization...")
    pipelines = optimizer.optimize()
    
    if pipelines:
        best_path, best_strategy, best_time = pipelines[0]
        print(f"  Best pipeline: {best_path}")
        print(f"  Best execution time: {best_time:.4f}s")
        
        # Print partition assignment
        print("  Partition assignments:")
        for v_id, partitions in best_strategy.items():
            if partitions:
                partition_names = [p.name for p in partitions]
                print(f"    Vehicle {v_id}: {', '.join(partition_names)}")
            else:
                print(f"    Vehicle {v_id}: No partitions assigned")
    else:
        print("  No valid pipelines found.")
    
    return optimizer

def main():
    print("Model Utility Example".center(80, "="))
    
    # 1. Create a model manually
    manual_model = create_manual_model()
    
    # 2. Create a model from an XML file
    xml_model = create_from_xml()
    
    # 3. Create a random model
    random_model = create_random_model()
    
    # 4. Modify and save a model
    modified_model = modify_and_save_model(random_model)
    
    # 5. Analyze model components and partitions
    analysis_results = analyze_model_and_partitions(xml_model)
    
    # 6. Use the model with an optimizer
    optimizer = use_model_with_optimizer(xml_model)
    
    print("\nAll model operations completed successfully!".center(80, "="))

if __name__ == "__main__":
    main()
