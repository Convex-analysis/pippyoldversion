#!/usr/bin/env python3
"""
FHDP Heterogeneous Platform Example

Demonstrates FHDP deployment across heterogeneous platforms including
Jetson Orin Nano, x86 PCs, and other devices.
"""
import sys
import time
import logging
from pathlib import Path

# Add FHDP to path
sys.path.append(str(Path(__file__).parent.parent))

from fhdp.core.hardware_adapter import (
    HardwareDetector, HardwarePlatform, ResourceAdapter,
    HardwareCapabilities
)
from fhdp.core.heterogeneous_resource import (
    AdaptiveResourceMonitor, HeterogeneousScheduler, ComputeWorkload, TaskComplexity
)
from core.cross_platform_comm import PlatformBridge, NetworkEndpoint, TransportProtocol
from core.load_balancer import IntelligentLoadBalancer, LoadBalancingStrategy, TaskInfo

def main():
    """Main heterogeneous platform example"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("🚀 FHDP Heterogeneous Platform Example")
    print("=" * 50)
    
    # 1. Detect current hardware platform
    logger.info("Step 1: Detecting hardware platform...")
    detector = HardwareDetector()
    platform = detector.detect_platform()
    
    print(f"📱 Detected Platform: {platform.value}")
    
    # Get detailed capabilities
    capabilities = detector.get_hardware_capabilities()
    print(f"🔧 Hardware Capabilities:")
    print(f"   Platform: {capabilities.platform.value}")
    print(f"   Compute Capability: {capabilities.compute_capability.value}")
    print(f"   CPU Cores: {capabilities.cpu_cores}")
    print(f"   CPU Frequency: {capabilities.cpu_freq:.1f} GHz")
    print(f"   Total Memory: {capabilities.memory_total:.1f} GB")
    print(f"   GPU Memory: {capabilities.gpu_memory:.1f} GB")
    print(f"   Network Speed: {capabilities.network_speed:.0f} Mbps")
    
    # 2. Start adaptive resource monitoring
    logger.info("Step 2: Starting adaptive resource monitoring...")
    resource_monitor = AdaptiveResourceMonitor(capabilities)
    resource_monitor.start_monitoring()
    
    print(f"📊 Started resource monitoring for {platform.value}")
    
    # Wait for initial monitoring data
    time.sleep(2)
    
    # 3. Start heterogeneous scheduler
    logger.info("Step 3: Starting heterogeneous scheduler...")
    scheduler = HeterogeneousScheduler(capabilities)
    scheduler.start_scheduler()
    
    print(f"⚙️  Started heterogeneous scheduler")
    
    # 4. Setup cross-platform communication
    logger.info("Step 4: Setting up cross-platform communication...")
    platform_bridge = PlatformBridge(capabilities)
    
    # Create network endpoint for communication
    network_endpoint = NetworkEndpoint(
        host="localhost",
        port=8080,
        protocol=TransportProtocol.TCP,
        compression=CompressionType.ZLIB
    )
    
    print(f"🌐 Set up cross-platform communication bridge")
    
    # 5. Setup intelligent load balancer
    logger.info("Step 5: Setting up intelligent load balancer...")
    load_balancer = IntelligentLoadBalancer(LoadBalancingStrategy.HYBRID)
    load_balancer.start_load_balancing()
    
    # Register current node
    load_balancer.register_node(
        node_id=f"local_{platform.value}",
        capabilities=capabilities,
        max_tasks=4,
        network_latency=1.0
    )
    
    print(f"⚖️  Started intelligent load balancer with HYBRID strategy")
    
    # 6. Create and submit example workloads
    logger.info("Step 6: Creating example workloads...")
    
    workloads = [
        ComputeWorkload(
            workload_id="light_inference_1",
            complexity=TaskComplexity.LIGHT,
            cpu_requirement=0.2,
            memory_requirement=0.1,
            gpu_requirement=0.1,
            duration_estimate=5.0,
            priority=5,
            platform_preferences=[platform]
        ),
        ComputeWorkload(
            workload_id="medium_training_1", 
            complexity=TaskComplexity.MEDIUM,
            cpu_requirement=0.5,
            memory_requirement=0.3,
            gpu_requirement=0.4,
            duration_estimate=15.0,
            priority=3,
            platform_preferences=[platform]
        ),
        ComputeWorkload(
            workload_id="heavy_computation_1",
            complexity=TaskComplexity.HEAVY,
            cpu_requirement=0.8,
            memory_requirement=0.6,
            gpu_requirement=0.7,
            duration_estimate=30.0,
            priority=2,
            platform_preferences=[platform]
        )
    ]
    
    # Submit workloads to scheduler
    print(f"📋 Submitting {len(workloads)} workloads...")
    for workload in workloads:
        scheduler.submit_workload(workload)
        
        # Create task info for load balancer
        task_info = TaskInfo(
            task_id=workload.workload_id,
            workload=workload,
            priority=workload.priority,
            estimated_duration=workload.duration_estimate
        )
        
        load_balancer.submit_task(task_info)
        
        print(f"   Submitted: {workload.workload_id} ({workload.complexity.value})")
    
    # 7. Monitor resource usage and performance
    logger.info("Step 7: Monitoring system performance...")
    
    monitor_duration = 30  # seconds
    monitor_interval = 5   # seconds
    
    print(f"⏱️  Monitoring for {monitor_duration} seconds...")
    print("\n" + "="*60)
    print("RESOURCE MONITORING DASHBOARD")
    print("="*60)
    
    for i in range(monitor_duration // monitor_interval):
        # Get current resource metrics
        current_metrics = resource_monitor.get_resource_metrics()
        predicted_metrics = resource_monitor.predict_resource_availability(10.0)
        
        print(f"\n📊 T+{i*monitor_interval}s - Resource Metrics:")
        print(f"   CPU Usage:        {current_metrics.cpu_usage*100:.1f}% → {predicted_metrics.cpu_usage*100:.1f}% (pred)")
        print(f"   Memory Usage:     {current_metrics.memory_usage*100:.1f}% → {predicted_metrics.memory_usage*100:.1f}% (pred)")
        print(f"   Battery Level:    {current_metrics.battery_level*100:.1f}% → {predicted_metrics.battery_level*100:.1f}% (pred)")
        print(f"   Network Quality:  {current_metrics.network_quality*100:.1f}%")
        print(f"   Thermal State:    {current_metrics.thermal_state*100:.1f}%")
        
        # Get load balancer stats
        lb_stats = load_balancer.get_load_balancer_stats()
        print(f"\n⚖️  Load Balancer Stats:")
        print(f"   Queue Size:       {lb_stats['queue_size']}")
        print(f"   Active Tasks:     {lb_stats['active_tasks']}")
        print(f"   Tasks Scheduled:  {lb_stats['statistics']['tasks_scheduled']}")
        print(f"   Tasks Completed:  {lb_stats['statistics']['tasks_completed']}")
        print(f"   Average Wait:     {lb_stats['statistics']['average_wait_time']:.2f}s")
        
        # Get scheduler stats
        scheduler_stats = scheduler.get_scheduler_stats()
        print(f"\n⚙️  Scheduler Stats:")
        print(f"   Queue Size:       {scheduler_stats['queue_size']}")
        print(f"   Success Rate:     {scheduler_stats['success_rate']*100:.1f}%")
        print(f"   Platform Distribution: {scheduler_stats['platform_distribution']}")
        
        # Get bridge status
        bridge_status = platform_bridge.get_bridge_status()
        print(f"\n🌐 Platform Bridge:")
        print(f"   Local Platform:  {bridge_status['local_platform']}")
        print(f"   Active Bridges:  {bridge_status['active_bridges']}")
        print(f"   Network Stats:    {bridge_status['network_stats']}")
        
        # Check if can handle different workload types
        print(f"\n🎯 Workload Capability Analysis:")
        test_complexities = [TaskComplexity.LIGHT, TaskComplexity.MEDIUM, 
                            TaskComplexity.HEAVY, TaskComplexity.EXTREME]
        
        for complexity in test_complexities:
            test_workload = ComputeWorkload(
                workload_id=f"test_{complexity.value}",
                complexity=complexity,
                cpu_requirement=0.5,
                memory_requirement=0.3,
                gpu_requirement=0.2,
                duration_estimate=10.0,
                priority=1
            )
            
            can_handle, reason, availability = resource_monitor.can_handle_workload(test_workload)
            status = "✅" if can_handle else "❌"
            print(f"   {status} {complexity.value.capitalize():10} | {reason} ({availability:.2f})")
        
        print("\n" + "-"*60)
        
        # Simulate some task completion
        if i == 2:
            load_balancer.complete_task("light_inference_1", True, 4.8)
            print("✅ Completed light_inference_1")
        elif i == 4:
            load_balancer.complete_task("medium_training_1", True, 14.2)
            print("✅ Completed medium_training_1")
        elif i == 6:
            load_balancer.complete_task("heavy_computation_1", True, 28.5)
            print("✅ Completed heavy_computation_1")
        
        time.sleep(monitor_interval)
    
    # 8. Platform-specific recommendations
    logger.info("Step 8: Generating platform-specific recommendations...")
    
    print(f"\n🎯 Platform-Specific Recommendations for {platform.value}:")
    
    if platform == HardwarePlatform.JETSON_ORIN:
        print("   🚀 Jetson Orin Optimizations:")
        print("      • Use TensorRT for inference acceleration")
        print("      • Enable max performance mode for heavy workloads")
        print("      • Optimize memory usage with unified memory")
        print("      • Leverage GPU for ML tasks")
        
    elif platform == HardwarePlatform.JETSON_NANO:
        print("   🔋 Jetson Nano Optimizations:")
        print("      • Use 10W power mode for balanced performance")
        print("      • Limit batch sizes to fit memory constraints")
        print("      • Prefer light inference tasks")
        print("      • Monitor thermal state closely")
        
    elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_WINDOWS]:
        print("   💻 x86 Platform Optimizations:")
        print("      • Use multi-threading for parallel processing")
        print("      • Leverage GPU acceleration when available")
        print("      • Optimize for heavy computation tasks")
        print("      • Use larger batch sizes for efficiency")
    
    # 9. Cleanup
    logger.info("Step 9: Cleaning up resources...")
    
    print(f"\n🧹 Shutting down services...")
    
    load_balancer.stop_load_balancing()
    scheduler.stop_scheduler()
    resource_monitor.stop_monitoring()
    
    # 10. Final statistics
    logger.info("Step 10: Final statistics...")
    
    print(f"\n📈 Final Statistics:")
    
    final_lb_stats = load_balancer.get_load_balancer_stats()
    print(f"   Load Balancer:")
    print(f"     Total Tasks Scheduled: {final_lb_stats['statistics']['tasks_scheduled']}")
    print(f"     Total Tasks Completed: {final_lb_stats['statistics']['tasks_completed']}")
    print(f"     Success Rate: {final_lb_stats['statistics']['success_rate']*100:.1f}%")
    print(f"     Average Wait Time: {final_lb_stats['statistics']['average_wait_time']:.2f}s")
    
    final_scheduler_stats = scheduler.get_scheduler_stats()
    print(f"   Scheduler:")
    print(f"     Success Rate: {final_scheduler_stats['success_rate']*100:.1f}%")
    print(f"     Platform Distribution: {final_scheduler_stats['platform_distribution']}")
    
    print(f"\n🎉 Heterogeneous platform example completed!")
    print(f"🏗️  FHDP successfully adapted to {platform.value} platform")
    
    # Platform-specific deployment instructions
    print(f"\n📋 Deployment Instructions for {platform.value}:")
    print(f"   1. Run: python scripts/deploy_heterogeneous.py --config config/heterogeneous_config.yaml --platform {platform.value.lower()}")
    print(f"   2. Start FHDP: ./fhdp_startup.sh")
    print(f"   3. Monitor: tail -f /var/log/fhdp/fhdp.log")

if __name__ == "__main__":
    main()