"""EdgePipe example usage"""

import numpy as np
from .super_neuron import SuperNeuronNetwork
from .partitioning import HybridPartitioning
from .device_mapping import NeuronDeviceMapping
from .pipeline_scheduler import PipelineScheduler
from .performance_analysis import PerformanceAnalyzer

def edgepipe_example():
    """
    EdgePipe example usage
    """
    # Example configuration
    total_layers = 6  # 1 input layer + 5 hidden layers
    total_devices = 4
    neurons_per_layer = [784, 128, 128, 128, 128, 10]  # MNIST-like model
    batch_size = 100
    
    print("=== EdgePipe Example ===")
    print(f"Total layers: {total_layers}")
    print(f"Total devices: {total_devices}")
    print(f"Neurons per layer: {neurons_per_layer}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Step 1: Perform hybrid partitioning
    print("Step 1: Performing hybrid partitioning...")
    partitioning = HybridPartitioning(total_layers, total_devices)
    super_neuron_network = partitioning.perform_partitioning(neurons_per_layer)
    
    # Print partitioning info
    partitioning_info = partitioning.get_partitioning_info()
    print(f"Partitioning info:")
    print(f"  M_layers: {partitioning_info['M_layers']}")
    print(f"  Layer groups: {partitioning_info['layer_groups']}")
    print(f"  Device allocation: {partitioning_info['device_allocation']}")
    print(f"  Total super neurons: {partitioning_info['total_super_neurons']}")
    print()
    
    # Print super neurons
    print("Super neurons created:")
    for sn in super_neuron_network.get_super_neurons():
        print(f"  {sn}")
    print()
    
    # Step 2: Create device network (PRR matrix)
    print("Step 2: Creating device network...")
    # Simulate PRR values between 0.7 and 1.0
    device_network = np.random.uniform(0.7, 1.0, (total_devices, total_devices))
    # Set diagonal to 1.0 (device to itself)
    np.fill_diagonal(device_network, 1.0)
    print("Device network (PRR matrix):")
    print(device_network)
    print()
    
    # Step 3: Optimize neuron to device mapping
    print("Step 3: Optimizing neuron to device mapping...")
    mapping = NeuronDeviceMapping(super_neuron_network, device_network)
    best_mapping = mapping.optimize_mapping(generations=1000)
    
    # Print mapping info
    mapping_info = mapping.get_mapping_info()
    print(f"Mapping info:")
    print(f"  Best mapping: {mapping_info['best_mapping']}")
    print(f"  Best score: {mapping_info['best_score']:.4f}")
    print(f"  Super neurons per device: {mapping_info['super_neurons_per_device']}")
    print()
    
    # Step 4: Generate pipeline schedule
    print("Step 4: Generating pipeline schedule...")
    scheduler = PipelineScheduler(super_neuron_network)
    schedule = scheduler.generate_schedule(batch_size)
    
    # Print pipeline info
    pipeline_info = scheduler.get_pipeline_info()
    print(f"Pipeline info:")
    print(f"  Pipeline stages: {pipeline_info['pipeline_stages']}")
    print(f"  Super neurons per device: {pipeline_info['super_neurons_per_device']}")
    print()
    
    # Step 5: Calculate execution time
    print("Step 5: Calculating execution time...")
    forward_time_per_layer = 0.01  # 10ms per layer
    backward_time_per_layer = 0.015  # 15ms per layer
    execution_time = scheduler.calculate_execution_time(forward_time_per_layer, backward_time_per_layer, batch_size)
    print(f"Estimated execution time: {execution_time:.4f} seconds")
    print()
    
    # Step 6: Analyze performance
    print("Step 6: Analyzing performance...")
    analyzer = PerformanceAnalyzer(super_neuron_network)
    performance_summary = analyzer.get_performance_summary(batch_size)
    
    print(f"Performance summary:")
    print(f"  Allocation type: {performance_summary['allocation_type']}")
    print(f"  Time complexity: {performance_summary['time_complexity']}")
    print(f"  Total slots: {performance_summary['total_slots']}")
    print(f"  Speedup: {performance_summary['speedup']:.4f}x")
    print()
    
    # Step 7: Analyze fault tolerance
    print("Step 7: Analyzing fault tolerance...")
    fault_tolerance = analyzer.analyze_fault_tolerance(0.1, batch_size)  # 10% failure probability
    print(f"Fault tolerance analysis:")
    print(f"  Failure probability: {fault_tolerance['failure_probability']:.2f}")
    print(f"  Total devices: {fault_tolerance['total_devices']}")
    print(f"  Success probability: {fault_tolerance['success_probability']:.4f}")
    print(f"  Expected successful batches: {fault_tolerance['expected_successful_batches']:.2f}")
    print()
    
    # Step 8: Analyze scalability
    print("Step 8: Analyzing scalability...")
    scalability = analyzer.analyze_scalability(8, neurons_per_layer)
    print("Scalability analysis:")
    print("  Device count | M_layers | Total SN | Time slots | Speedup")
    print("  " + "-" * 60)
    for result in scalability['scalability_results']:
        print(f"  {result['device_count']:11d} | {result['M_layers']:7d} | {result['total_super_neurons']:8d} | {result['time_slots']:10d} | {result['speedup']:7.2f}x")
    print()
    
    print("=== EdgePipe Example Complete ===")

if __name__ == "__main__":
    edgepipe_example()
