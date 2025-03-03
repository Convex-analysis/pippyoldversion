import xml.etree.ElementTree as ET
import os
import numpy as np
from model import ModelComponent, UnitModelPartition, DNNModel

def load_model_from_xml(xml_path):
    """
    Load a DNN model configuration from an XML file.
    
    The XML format should be:
    
    <model name="ModelName">
        <components>
            <component name="RGB_Backbone" flops_per_sample="5e9" capacity="2e9" />
            <component name="Lidar_Backbone" flops_per_sample="8e9" capacity="3e9" />
            <!-- more components... -->
        </components>
        <partitions>
            <partition name="RGB_Only" communication_volume="0.5e9">
                <component_ref name="RGB_Backbone" />
            </partition>
            <partition name="Lidar_Only" communication_volume="0.8e9">
                <component_ref name="Lidar_Backbone" />
            </partition>
            <!-- more partitions... -->
        </partitions>
    </model>
    
    Args:
        xml_path: Path to the XML file containing model configuration
        
    Returns:
        DNNModel object with components and partitions
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
    
    # Get model name
    model_name = root.get('name', 'UnnamedModel')
    model = DNNModel(model_name)
    
    # Parse components
    components_elem = root.find('components')
    if components_elem is None:
        raise ValueError("No components section found in XML")
    
    # Dictionary to store components for later reference in partitions
    component_dict = {}
    
    for comp_elem in components_elem.findall('component'):
        # Get component attributes
        name = comp_elem.get('name')
        if name is None:
            raise ValueError("Component missing required 'name' attribute")
        
        try:
            # Convert values from scientific notation strings to float
            flops_per_sample = float(eval(comp_elem.get('flops_per_sample', '0')))
            capacity = float(eval(comp_elem.get('capacity', '0')))
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid numeric value for component {name}: {e}")
        
        # Create component and add to model
        component = ModelComponent(name, flops_per_sample, capacity)
        model.add_component(component)
        component_dict[name] = component
    
    # Parse partitions
    partitions_elem = root.find('partitions')
    if partitions_elem is not None:
        for part_elem in partitions_elem.findall('partition'):
            # Get partition attributes
            name = part_elem.get('name')
            if name is None:
                raise ValueError("Partition missing required 'name' attribute")
            
            try:
                # Convert communication volume from string to float
                communication_volume = float(eval(part_elem.get('communication_volume', '0')))
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Invalid numeric value for partition {name}: {e}")
            
            # Get components referenced in this partition
            components = []
            for comp_ref in part_elem.findall('component_ref'):
                comp_name = comp_ref.get('name')
                if comp_name not in component_dict:
                    raise ValueError(f"Partition {name} references unknown component: {comp_name}")
                components.append(component_dict[comp_name])
            
            if not components:
                raise ValueError(f"Partition {name} does not reference any components")
            
            # Calculate total FLOPS and capacity for this partition
            total_flops = sum(comp.flops_per_sample for comp in components)
            total_capacity = sum(comp.capacity for comp in components)
            
            # Create partition and add to model
            partition = UnitModelPartition(name, components, total_flops, total_capacity, communication_volume)
            model.add_unit_partition(partition)
    
    return model

def save_model_to_xml(model, xml_path):
    """
    Save a DNN model configuration to an XML file.
    
    Args:
        model: DNNModel object
        xml_path: Path to save the XML file
    """
    # Create root element
    root = ET.Element('model')
    root.set('name', model.name)
    
    # Add components section
    components_elem = ET.SubElement(root, 'components')
    for component in model.components:
        comp_elem = ET.SubElement(components_elem, 'component')
        comp_elem.set('name', component.name)
        comp_elem.set('flops_per_sample', str(component.flops_per_sample))
        comp_elem.set('capacity', str(component.capacity))
    
    # Add partitions section if there are any partitions
    if model.unit_partitions:
        partitions_elem = ET.SubElement(root, 'partitions')
        for partition in model.unit_partitions:
            part_elem = ET.SubElement(partitions_elem, 'partition')
            part_elem.set('name', partition.name)
            part_elem.set('communication_volume', str(partition.communication_volume))
            
            # Add component references
            for component in partition.components:
                comp_ref = ET.SubElement(part_elem, 'component_ref')
                comp_ref.set('name', component.name)
    
    # Create XML tree and write to file
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"Model configuration saved to: {xml_path}")

def generate_random_model(name, num_components=5, num_partitions=3, 
                         min_flops=1e9, max_flops=10e9, 
                         min_capacity=0.5e9, max_capacity=4e9,
                         min_comm=0.2e9, max_comm=2e9, 
                         seed=None):
    """
    Generate a random DNN model with components and partitions.
    
    Args:
        name: Model name
        num_components: Number of components to generate
        num_partitions: Number of partitions to generate
        min_flops, max_flops: Range for FLOPS requirements
        min_capacity, max_capacity: Range for capacity requirements
        min_comm, max_comm: Range for communication volume
        seed: Random seed for reproducibility
        
    Returns:
        DNNModel object with randomly generated components and partitions
    """
    # Set random seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
    
    # Create model
    model = DNNModel(name)
    
    # Create components
    components = []
    for i in range(num_components):
        comp_name = f"Component_{i+1}"
        flops = np.random.uniform(min_flops, max_flops)
        capacity = np.random.uniform(min_capacity, max_capacity)
        
        component = ModelComponent(comp_name, flops, capacity)
        model.add_component(component)
        components.append(component)
    
    # Create partitions
    # Ensure all components are assigned to at least one partition
    # Start with one component per partition
    if num_partitions > num_components:
        num_partitions = num_components  # Can't have more partitions than components
        
    components_copy = components.copy()
    np.random.shuffle(components_copy)
    
    for i in range(num_partitions):
        part_name = f"Partition_{i+1}"
        
        # Start with one component
        if i < len(components_copy):
            partition_components = [components_copy[i]]
        else:
            # If we have more partitions than components, reuse components
            partition_components = [np.random.choice(components)]
        
        # Randomly add more components with 50% chance each
        for comp in components:
            if comp not in partition_components and np.random.random() < 0.3:
                partition_components.append(comp)
        
        # Calculate total flops and capacity
        total_flops = sum(c.flops_per_sample for c in partition_components)
        total_capacity = sum(c.capacity for c in partition_components)
        
        # Randomly set communication volume
        comm_volume = np.random.uniform(min_comm, max_comm)
        
        # Create partition
        partition = UnitModelPartition(part_name, partition_components, total_flops, total_capacity, comm_volume)
        model.add_unit_partition(partition)
    
    return model

def model_to_graphviz(model, output_path=None, show_components=True):
    """
    Create a graphical representation of the model structure using graphviz.
    
    Args:
        model: DNNModel object
        output_path: Optional path to save the visualization
        show_components: Whether to show component details in the graph
        
    Returns:
        Graphviz object representing the model structure
    """
    try:
        import graphviz
    except ImportError:
        print("Warning: graphviz package not installed. Installing it with:")
        print("  pip install graphviz")
        print("And ensure the Graphviz binary is installed on your system.")
        return None
    
    # Create a new graph
    dot = graphviz.Digraph(comment=f'Model: {model.name}')
    dot.attr(rankdir='TB', size='11,11', dpi='300')
    
    # Add partition nodes
    with dot.subgraph(name='cluster_partitions') as p:
        p.attr(label='Partitions', style='filled', color='lightgrey')
        for i, partition in enumerate(model.unit_partitions):
            flops_gb = partition.flops_per_sample / 1e9
            capacity_gb = partition.capacity / 1e9
            comm_gb = partition.communication_volume / 1e9
            
            p.node(
                f'p{i}', 
                f'{partition.name}\nFLOPs: {flops_gb:.2f} G\nCapacity: {capacity_gb:.2f} GB\nComm: {comm_gb:.2f} GB',
                shape='box', style='filled', fillcolor='lightblue'
            )
    
    # Add component nodes if requested
    if show_components:
        with dot.subgraph(name='cluster_components') as c:
            c.attr(label='Components', style='filled', color='lightgrey')
            for i, component in enumerate(model.components):
                flops_gb = component.flops_per_sample / 1e9
                capacity_gb = component.capacity / 1e9
                
                c.node(
                    f'c{i}', 
                    f'{component.name}\nFLOPs: {flops_gb:.2f} G\nCapacity: {capacity_gb:.2f} GB',
                    shape='ellipse', style='filled', fillcolor='lightyellow'
                )
        
        # Add edges from partitions to components
        for i, partition in enumerate(model.unit_partitions):
            for component in partition.components:
                j = model.components.index(component)
                dot.edge(f'p{i}', f'c{j}', style='dashed')
    
    # Render the graph
    if output_path:
        try:
            dot.render(output_path, format='png', cleanup=True)
            print(f"Model visualization saved to: {output_path}.png")
        except Exception as e:
            print(f"Error rendering graph: {e}")
    
    return dot

# Example usage
if __name__ == "__main__":
    # Generate a random model
    random_model = generate_random_model(
        "RandomModel", 
        num_components=7, 
        num_partitions=4,
        seed=42
    )
    
    # Save to XML
    save_model_to_xml(random_model, "random_model.xml")
    
    # Load from XML
    loaded_model = load_model_from_xml("random_model.xml")
    
    # Print model information
    print(f"Model: {loaded_model.name}")
    print(f"Components ({len(loaded_model.components)}):")
    for i, component in enumerate(loaded_model.components):
        print(f"  {i+1}. {component.name}: {component.flops_per_sample/1e9:.2f} GFLOPS, {component.capacity/1e9:.2f} GB")
    
    print(f"\nPartitions ({len(loaded_model.unit_partitions)}):")
    for i, partition in enumerate(loaded_model.unit_partitions):
        component_names = [c.name for c in partition.components]
        print(f"  {i+1}. {partition.name}: {partition.flops_per_sample/1e9:.2f} GFLOPS, " +
              f"{partition.capacity/1e9:.2f} GB, {partition.communication_volume/1e9:.2f} GB comm")
        print(f"     Components: {', '.join(component_names)}")
    
    # Create visualization
    model_to_graphviz(loaded_model, "random_model")
