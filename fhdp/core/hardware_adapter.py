"""
Hardware Adaptation Layer for FHDP System

Provides abstraction for different hardware platforms including Jetson Orin Nano,
x86 PCs, and other heterogeneous computing devices.
"""
import platform
import subprocess
import socket
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np

# Try to import psutil for system resource monitoring
psutil_available = False
try:
    import psutil
    psutil_available = True
except ImportError:
    pass

from ..core.types import ResourceMetrics, ResourceClass

class HardwarePlatform(Enum):
    """Supported hardware platforms"""
    JETSON_ORIN = "jetson_orin"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    X86_LINUX = "x86_linux"
    X86_WINDOWS = "x86_windows"
    X86_MACOS = "x86_macos"
    ARM_LINUX = "arm_linux"
    UNKNOWN = "unknown"

class ComputeCapability(Enum):
    """Compute capability levels"""
    EDGE_AI = "edge_ai"  # Jetson Orin/Nano - strong GPU/NPU
    SERVER_CLASS = "server_class"  # x86 workstation
    EMBEDDED = "embedded"  # ARM devices
    BASIC = "basic"  # Minimal compute devices

@dataclass
class HardwareCapabilities:
    """Hardware capability specification"""
    platform: HardwarePlatform
    compute_capability: ComputeCapability
    cpu_cores: int
    cpu_freq: float  # GHz
    memory_total: float  # GB
    gpu_memory: float  # GB (0 if no GPU)
    npu_memory: float  # GB (0 if no NPU)
    storage_speed: str  # 'emmc', 'ssd', 'hdd'
    network_speed: float  # Mbps
    power_profile: str  # 'high_performance', 'balanced', 'power_saver'
    thermal_limit: float  # °C
    accelerated_compute: bool  # CUDA/OpenCL/Vulkan support

@dataclass
class NetworkInterface:
    """Network interface information"""
    name: str
    type: str  # 'ethernet', 'wifi', 'cellular'
    speed: float  # Mbps
    latency: float  # ms (estimated)
    reliable: bool  # Stable connection?
    ipv4: str
    ipv6: Optional[str] = None

class HardwareDetector:
    """Detects and profiles hardware capabilities"""
    
    def __init__(self):
        self.platform_cache = {}
        self.capabilities_cache = {}
        self.cache_timeout = 300.0  # 5 minutes
        
    def detect_platform(self) -> HardwarePlatform:
        """Detect the current hardware platform"""
        system = platform.system()
        machine = platform.machine()
        
        # Check for Jetson devices
        if system == "Linux":
            try:
                # Read Jetson-specific files
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip().lower()
                    
                if 'orin' in model:
                    return HardwarePlatform.JETSON_ORIN
                elif 'nano' in model:
                    return HardwarePlatform.JETSON_NANO
                elif 'xavier' in model:
                    return HardwarePlatform.JETSON_XAVIER
                    
            except (FileNotFoundError, PermissionError):
                pass
            
            # Check for ARM Linux
            if machine in ('armv7l', 'aarch64', 'arm64'):
                return HardwarePlatform.ARM_LINUX
            else:
                return HardwarePlatform.X86_LINUX
        elif system == "Windows":
            return HardwarePlatform.X86_WINDOWS
        elif system == "Darwin":
            return HardwarePlatform.X86_MACOS
        else:
            return HardwarePlatform.UNKNOWN
    
    def get_hardware_capabilities(self) -> HardwareCapabilities:
        """Get detailed hardware capabilities"""
        platform = self.detect_platform()
        
        # Check cache
        cache_key = f"{platform}_{time.time() // self.cache_timeout}"
        if cache_key in self.capabilities_cache:
            return self.capabilities_cache[cache_key]
        
        capabilities = self._profile_hardware(platform)
        
        # Cache result
        self.capabilities_cache[cache_key] = capabilities
        return capabilities
    
    def _profile_hardware(self, platform: HardwarePlatform) -> HardwareCapabilities:
        """Profile hardware based on platform"""
        
        # Get basic system info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq().current / 1000.0 if psutil.cpu_freq() else 2.0
        memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        # Platform-specific profiling
        if platform == HardwarePlatform.JETSON_ORIN:
            return self._profile_jetson_orin(cpu_count, cpu_freq, memory)
        elif platform == HardwarePlatform.JETSON_NANO:
            return self._profile_jetson_nano(cpu_count, cpu_freq, memory)
        elif platform == HardwarePlatform.JETSON_XAVIER:
            return self._profile_jetson_xavier(cpu_count, cpu_freq, memory)
        elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_WINDOWS, HardwarePlatform.X86_MACOS]:
            return self._profile_x86(platform, cpu_count, cpu_freq, memory)
        else:
            return self._profile_generic(platform, cpu_count, cpu_freq, memory)
    
    def _profile_jetson_orin(self, cpu_count: int, cpu_freq: float, memory: float) -> HardwareCapabilities:
        """Profile Jetson Orin device"""
        try:
            # Read Jetson clocks
            with open('/sys/kernel/debug/clk/cluster0/clk_rate', 'r') as f:
                cluster0_freq = int(f.read().strip()) / 1e9
                
            # Get GPU info
            gpu_memory = self._get_jetson_gpu_memory()
            
            # Check for NPU (some Orin models have NPU)
            npu_memory = self._get_jetson_npu_memory()
            
            return HardwareCapabilities(
                platform=HardwarePlatform.JETSON_ORIN,
                compute_capability=ComputeCapability.EDGE_AI,
                cpu_cores=cpu_count,
                cpu_freq=cluster0_freq,
                memory_total=memory,
                gpu_memory=gpu_memory,
                npu_memory=npu_memory,
                storage_speed='emmc',  # Most Jetsons use eMMC
                network_speed=1000.0,  # Gigabit Ethernet
                power_profile='high_performance',
                thermal_limit=85.0,
                accelerated_compute=True
            )
        except Exception as e:
            print(f"Error profiling Jetson Orin: {e}")
            # Fallback to generic values
            return HardwareCapabilities(
                platform=HardwarePlatform.JETSON_ORIN,
                compute_capability=ComputeCapability.EDGE_AI,
                cpu_cores=cpu_count,
                cpu_freq=cpu_freq,
                memory_total=memory,
                gpu_memory=8.0,  # Typical Orin GPU memory
                npu_memory=2.0,  # Typical Orin NPU memory
                storage_speed='emmc',
                network_speed=1000.0,
                power_profile='high_performance',
                thermal_limit=85.0,
                accelerated_compute=True
            )
    
    def _profile_jetson_nano(self, cpu_count: int, cpu_freq: float, memory: float) -> HardwareCapabilities:
        """Profile Jetson Nano device"""
        return HardwareCapabilities(
            platform=HardwarePlatform.JETSON_NANO,
            compute_capability=ComputeCapability.EDGE_AI,
            cpu_cores=cpu_count,
            cpu_freq=cpu_freq,
            memory_total=memory,
            gpu_memory=2.0,  # Nano has 2GB GPU memory
            npu_memory=0.0,
            storage_speed='emmc',
            network_speed=100.0,  # Usually 100Mbps on Nano
            power_profile='balanced',
            thermal_limit=75.0,
            accelerated_compute=True
        )
    
    def _profile_jetson_xavier(self, cpu_count: int, cpu_freq: float, memory: float) -> HardwareCapabilities:
        """Profile Jetson Xavier device"""
        return HardwareCapabilities(
            platform=HardwarePlatform.JETSON_XAVIER,
            compute_capability=ComputeCapability.EDGE_AI,
            cpu_cores=cpu_count,
            cpu_freq=cpu_freq,
            memory_total=memory,
            gpu_memory=16.0,  # Xavier has 16GB GPU memory
            npu_memory=2.0,
            storage_speed='emmc',
            network_speed=10000.0,  # Xavier supports 10GbE
            power_profile='high_performance',
            thermal_limit=90.0,
            accelerated_compute=True
        )
    
    def _profile_x86(self, platform: HardwarePlatform, cpu_count: int, cpu_freq: float, memory: float) -> HardwareCapabilities:
        """Profile x86 platform"""
        gpu_memory = self._get_x86_gpu_memory()
        
        # Determine storage speed
        storage_speed = 'hdd'
        try:
            disk_usage = psutil.disk_usage('/')
            # Check if it's SSD (simplified heuristic)
            if 'nvme' in str(psutil.disk_partitions()) or 'ssd' in str(psutil.disk_partitions()).lower():
                storage_speed = 'ssd'
        except:
            pass
        
        # Network speed detection
        network_speed = self._detect_network_speed()
        
        return HardwareCapabilities(
            platform=platform,
            compute_capability=ComputeCapability.SERVER_CLASS,
            cpu_cores=cpu_count,
            cpu_freq=cpu_freq,
            memory_total=memory,
            gpu_memory=gpu_memory,
            npu_memory=0.0,
            storage_speed=storage_speed,
            network_speed=network_speed,
            power_profile='high_performance',
            thermal_limit=95.0,
            accelerated_compute=gpu_memory > 0
        )
    
    def _profile_generic(self, platform: HardwarePlatform, cpu_count: int, cpu_freq: float, memory: float) -> HardwareCapabilities:
        """Profile generic/unknown platform"""
        return HardwareCapabilities(
            platform=platform,
            compute_capability=ComputeCapability.BASIC,
            cpu_cores=cpu_count,
            cpu_freq=cpu_freq,
            memory_total=memory,
            gpu_memory=0.0,
            npu_memory=0.0,
            storage_speed='unknown',
            network_speed=100.0,
            power_profile='balanced',
            thermal_limit=70.0,
            accelerated_compute=False
        )
    
    def _get_jetson_gpu_memory(self) -> float:
        """Get GPU memory on Jetson device"""
        try:
            result = subprocess.run(['tegrastats', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse tegrastats output for GPU memory
                # This is a simplified implementation
                return 8.0  # Default for Orin
        except:
            pass
        return 0.0
    
    def _get_jetson_npu_memory(self) -> float:
        """Get NPU memory on Jetson device"""
        # NPU detection is platform-specific
        # For now, return 0 (no NPU)
        return 0.0
    
    def _get_x86_gpu_memory(self) -> float:
        """Get GPU memory on x86 platform"""
        try:
            # Try NVIDIA GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                memory_mb = float(result.stdout.strip().split()[0])
                return memory_mb / 1024.0  # Convert to GB
        except:
            pass
        
        # Try AMD GPU (rocm)
        try:
            result = subprocess.run(['rocm-smi', '--showmeminfo'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse ROCm output
                return 4.0  # Default estimate
        except:
            pass
        
        return 0.0
    
    def _detect_network_speed(self) -> float:
        """Detect network interface speed"""
        try:
            net_io = psutil.net_io_counters()
            # This is a simplified approach
            # In real implementation, would use platform-specific APIs
            return 1000.0  # Default to 1Gbps
        except:
            return 100.0
    
    def get_network_interfaces(self) -> List[NetworkInterface]:
        """Get available network interfaces"""
        interfaces = []
        
        try:
            net_if_addrs = psutil.net_if_addrs()
            net_if_stats = psutil.net_if_stats()
            
            for name, addrs in net_if_addrs.items():
                stats = net_if_stats.get(name)
                if not stats:
                    continue
                
                # Get IP addresses
                ipv4 = None
                ipv6 = None
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ipv4 = addr.address
                    elif addr.family == socket.AF_INET6:
                        ipv6 = addr.address
                
                if ipv4:  # Only include interfaces with IPv4
                    interface_type = self._infer_interface_type(name)
                    speed = self._infer_interface_speed(name, interface_type)
                    
                    interfaces.append(NetworkInterface(
                        name=name,
                        type=interface_type,
                        speed=speed,
                        latency=1.0 if interface_type == 'ethernet' else 5.0,
                        reliable=stats.isup,
                        ipv4=ipv4,
                        ipv6=ipv6
                    ))
        except Exception as e:
            print(f"Error detecting network interfaces: {e}")
        
        return interfaces
    
    def _infer_interface_type(self, name: str) -> str:
        """Infer network interface type from name"""
        name_lower = name.lower()
        if 'eth' in name_lower or 'en' in name_lower:
            return 'ethernet'
        elif 'wifi' in name_lower or 'wl' in name_lower:
            return 'wifi'
        elif 'cellular' in name_lower or 'wwan' in name_lower:
            return 'cellular'
        else:
            return 'ethernet'  # Default assumption
    
    def _infer_interface_speed(self, name: str, interface_type: str) -> float:
        """Infer network interface speed"""
        if interface_type == 'ethernet':
            return 1000.0  # Assume 1Gbps for ethernet
        elif interface_type == 'wifi':
            return 300.0  # Assume 300Mbps for wifi
        else:
            return 100.0  # Default assumption

class ResourceAdapter:
    """Adapts resource monitoring to different hardware platforms"""
    
    def __init__(self):
        self.detector = HardwareDetector()
        self.capabilities = self.detector.get_hardware_capabilities()
        self.adapters = {
            HardwarePlatform.JETSON_ORIN: JetsonResourceAdapter,
            HardwarePlatform.JETSON_NANO: JetsonResourceAdapter,
            HardwarePlatform.JETSON_XAVIER: JetsonResourceAdapter,
            HardwarePlatform.X86_LINUX: X86ResourceAdapter,
            HardwarePlatform.X86_WINDOWS: X86ResourceAdapter,
            HardwarePlatform.X86_MACOS: X86ResourceAdapter,
            HardwarePlatform.ARM_LINUX: ARMResourceAdapter,
            HardwarePlatform.UNKNOWN: GenericResourceAdapter
        }
        
        self.adapter = self.adapters.get(self.capabilities.platform, GenericResourceAdapter)()
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics adapted for platform"""
        return self.adapter.get_current_metrics(self.capabilities)
    
    def predict_resources(self, horizon: float = 10.0) -> ResourceMetrics:
        """Predict future resource availability"""
        return self.adapter.predict_resources(self.capabilities, horizon)
    
    def classify_hardware(self) -> ResourceClass:
        """Classify hardware for task allocation"""
        return self.adapter.classify_hardware(self.capabilities)
    
    def get_platform_specific_info(self) -> Dict[str, Any]:
        """Get platform-specific information"""
        return self.adapter.get_platform_specific_info(self.capabilities)

class JetsonResourceAdapter:
    """Resource adapter for Jetson platforms"""
    
    def get_current_metrics(self, capabilities: HardwareCapabilities) -> ResourceMetrics:
        """Get current resource metrics for Jetson"""
        # Use Jetson-specific tools if available
        try:
            # Read Jetson-specific metrics
            with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
                temp_millidegrees = int(f.read().strip())
                thermal_state = min(1.0, temp_millidegrees / 1000.0 / capabilities.thermal_limit)
        except:
            thermal_state = 0.5
        
        # Standard psutil metrics
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory = psutil.virtual_memory()
        memory_usage = memory.used / memory.total
        
        # Battery (if available)
        try:
            battery = psutil.sensors_battery()
            battery_level = battery.percent / 100.0 if battery else 0.8
        except:
            battery_level = 0.8
        
        # Network quality estimate
        network_quality = self._estimate_jetson_network_quality()
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            battery_level=battery_level,
            network_quality=network_quality,
            thermal_state=thermal_state
        )
    
    def predict_resources(self, capabilities: HardwareCapabilities, horizon: float) -> ResourceMetrics:
        """Predict resources for Jetson"""
        current = self.get_current_metrics(capabilities)
        
        # Jetson-specific thermal prediction (tends to heat up under load)
        thermal_trend = 0.02 if current.thermal_state > 0.7 else 0.01
        predicted_thermal = min(1.0, current.thermal_state + thermal_trend * horizon)
        
        return ResourceMetrics(
            cpu_usage=current.cpu_usage,
            memory_usage=current.memory_usage,
            battery_level=current.battery_level - (0.001 * horizon),  # Battery drain
            network_quality=current.network_quality,
            thermal_state=predicted_thermal
        )
    
    def classify_hardware(self, capabilities: HardwareCapabilities) -> ResourceClass:
        """Classify Jetson hardware"""
        if capabilities.platform == HardwarePlatform.JETSON_ORIN:
            if capabilities.gpu_memory >= 8.0:
                return ResourceClass.HIGH
            else:
                return ResourceClass.MEDIUM
        elif capabilities.platform == HardwarePlatform.JETSON_XAVIER:
            return ResourceClass.HIGH
        else:  # Jetson Nano
            return ResourceClass.MEDIUM
    
    def get_platform_specific_info(self, capabilities: HardwareCapabilities) -> Dict[str, Any]:
        """Get Jetson-specific information"""
        return {
            'jetson_power_mode': self._get_jetson_power_mode(),
            'cuda_version': self._get_cuda_version(),
            'jetpack_version': self._get_jetpack_version(),
            'nvp_model': self._get_nvp_model(),
            'gpu_utilization': self._get_gpu_utilization(),
            'power_usage': self._get_power_usage()
        }
    
    def _estimate_jetson_network_quality(self) -> float:
        """Estimate network quality on Jetson"""
        # Jetson devices typically have good wired networking
        return 0.9
    
    def _get_jetson_power_mode(self) -> str:
        """Get current Jetson power mode"""
        try:
            with open('/sys/devices/platform/tegra-fan/target_pwm', 'r') as f:
                return f.read().strip()
        except:
            return "unknown"
    
    def _get_cuda_version(self) -> str:
        """Get CUDA version"""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.stdout.split()[-1] if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def _get_jetpack_version(self) -> str:
        """Get JetPack version"""
        try:
            result = subprocess.run(['cat', '/etc/nv_tegra_release'], 
                                  capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def _get_nvp_model(self) -> str:
        """Get NVP model string"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                return f.read().strip()
        except:
            return "unknown"
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization"""
        try:
            result = subprocess.run(['tegrastats', '--logfile', '/tmp/tegrastats.log', 
                                   '--interval', '100'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # Parse tegrastats output for GPU utilization
                return 0.5  # Placeholder
        except:
            pass
        return 0.0
    
    def _get_power_usage(self) -> float:
        """Get current power usage"""
        try:
            result = subprocess.run(['cat', '/sys/class/power_supply/battery/power_now'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                power_uw = int(result.stdout.strip())
                return power_uw / 1e6  # Convert to watts
        except:
            pass
        return 0.0

class X86ResourceAdapter:
    """Resource adapter for x86 platforms"""
    
    def get_current_metrics(self, capabilities: HardwareCapabilities) -> ResourceMetrics:
        """Get current resource metrics for x86"""
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory = psutil.virtual_memory()
        memory_usage = memory.used / memory.total
        
        # Battery (laptop) or default (desktop)
        try:
            battery = psutil.sensors_battery()
            battery_level = battery.percent / 100.0 if battery else 1.0
        except:
            battery_level = 1.0
        
        # Network quality based on interfaces
        network_quality = self._estimate_network_quality()
        
        # Thermal state estimation
        thermal_state = self._estimate_thermal_state()
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            battery_level=battery_level,
            network_quality=network_quality,
            thermal_state=thermal_state
        )
    
    def predict_resources(self, capabilities: HardwareCapabilities, horizon: float) -> ResourceMetrics:
        """Predict resources for x86"""
        current = self.get_current_metrics(capabilities)
        
        # x86 platforms generally have better thermal management
        return ResourceMetrics(
            cpu_usage=current.cpu_usage,
            memory_usage=current.memory_usage,
            battery_level=max(0.0, current.battery_level - 0.0005 * horizon),  # Slower drain
            network_quality=current.network_quality,
            thermal_state=current.thermal_state
        )
    
    def classify_hardware(self, capabilities: HardwareCapabilities) -> ResourceClass:
        """Classify x86 hardware"""
        if capabilities.cpu_cores >= 8 and capabilities.memory_total >= 16:
            return ResourceClass.HIGH
        elif capabilities.cpu_cores >= 4 and capabilities.memory_total >= 8:
            return ResourceClass.MEDIUM
        else:
            return ResourceClass.LOW
    
    def get_platform_specific_info(self, capabilities: HardwareCapabilities) -> Dict[str, Any]:
        """Get x86-specific information"""
        return {
            'os_info': f"{platform.system()} {platform.release()}",
            'cpu_model': platform.processor(),
            'gpu_info': self._get_gpu_info(),
            'virtualization': self._check_virtualization(),
            'hyperthreading': psutil.cpu_count(logical=False) < psutil.cpu_count(logical=True)
        }
    
    def _estimate_network_quality(self) -> float:
        """Estimate network quality"""
        try:
            net_io = psutil.net_io_counters()
            # Simple heuristic based on bytes sent/received
            total_bytes = net_io.bytes_sent + net_io.bytes_recv
            return min(1.0, total_bytes / 1e9)  # Normalize to GB
        except:
            return 0.8
    
    def _estimate_thermal_state(self) -> float:
        """Estimate thermal state on x86"""
        try:
            # Try to read temperature sensors
            temps = psutil.sensors_temperatures()
            if temps:
                # Use first available temperature sensor
                for name, entries in temps.items():
                    if entries and entries[0].current:
                        temp = entries[0].current
                        # Assume max safe temperature of 90°C
                        return min(1.0, temp / 90.0)
        except:
            pass
        return 0.3  # Default cool state
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        gpu_info = {}
        
        # NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                gpu_info['nvidia'] = {
                    'name': parts[0],
                    'memory_mb': int(parts[1]) if len(parts) > 1 else 0
                }
        except:
            pass
        
        # AMD GPU
        try:
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_info['amd'] = {'name': result.stdout.strip()}
        except:
            pass
        
        return gpu_info
    
    def _check_virtualization(self) -> bool:
        """Check if running in virtualized environment"""
        try:
            # Check for hypervisor indicators
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                return any(keyword in cpuinfo for keyword in ['hypervisor', 'vmware', 'virtualbox', 'kvm'])
        except:
            return False

class ARMResourceAdapter:
    """Resource adapter for ARM Linux platforms"""
    
    def get_current_metrics(self, capabilities: HardwareCapabilities) -> ResourceMetrics:
        """Get current resource metrics for ARM"""
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory = psutil.virtual_memory()
        memory_usage = memory.used / memory.total
        
        # ARM devices are often embedded with limited battery
        battery_level = 0.7  # Default assumption
        
        # Network quality
        network_quality = 0.7  # Often wireless
        
        # Thermal state (ARM devices can thermal throttle)
        thermal_state = self._estimate_arm_thermal_state()
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            battery_level=battery_level,
            network_quality=network_quality,
            thermal_state=thermal_state
        )
    
    def predict_resources(self, capabilities: HardwareCapabilities, horizon: float) -> ResourceMetrics:
        """Predict resources for ARM"""
        current = self.get_current_metrics(capabilities)
        return ResourceMetrics(
            cpu_usage=current.cpu_usage,
            memory_usage=current.memory_usage,
            battery_level=max(0.0, current.battery_level - 0.002 * horizon),  # Faster drain
            network_quality=current.network_quality,
            thermal_state=min(1.0, current.thermal_state + 0.01 * horizon)
        )
    
    def classify_hardware(self, capabilities: HardwareCapabilities) -> ResourceClass:
        """Classify ARM hardware"""
        if capabilities.memory_total >= 4 and capabilities.cpu_cores >= 4:
            return ResourceClass.MEDIUM
        else:
            return ResourceClass.LOW
    
    def get_platform_specific_info(self, capabilities: HardwareCapabilities) -> Dict[str, Any]:
        """Get ARM-specific information"""
        return {
            'architecture': platform.machine(),
            'big_little': self._detect_big_little(),
            'soc_info': self._get_soc_info()
        }
    
    def _estimate_arm_thermal_state(self) -> float:
        """Estimate thermal state for ARM"""
        try:
            # Try to read thermal zones
            thermal_zones = glob.glob('/sys/class/thermal/thermal_zone*/temp')
            temps = []
            for zone in thermal_zones:
                try:
                    with open(zone, 'r') as f:
                        temp_millidegrees = int(f.read().strip())
                        temps.append(temp_millidegrees / 1000.0)
                except:
                    continue
            
            if temps:
                avg_temp = sum(temps) / len(temps)
                return min(1.0, avg_temp / 85.0)  # Assume 85°C max
        except:
            pass
        return 0.4
    
    def _detect_big_little(self) -> bool:
        """Detect if using big.LITTLE architecture"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                # Look for different CPU frequencies in same system
                freq_lines = [line for line in cpuinfo.split('\n') if 'cpu MHz' in line]
                return len(set(line.split(':')[-1].strip() for line in freq_lines)) > 1
        except:
            return False
    
    def _get_soc_info(self) -> str:
        """Get SoC information"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                return f.read().strip()
        except:
            return "unknown"

class GenericResourceAdapter:
    """Generic resource adapter for unknown platforms"""
    
    def get_current_metrics(self, capabilities: HardwareCapabilities) -> ResourceMetrics:
        """Get current resource metrics using generic approach"""
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory = psutil.virtual_memory()
        memory_usage = memory.used / memory.total
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            battery_level=0.8,
            network_quality=0.7,
            thermal_state=0.4
        )
    
    def predict_resources(self, capabilities: HardwareCapabilities, horizon: float) -> ResourceMetrics:
        """Generic resource prediction"""
        current = self.get_current_metrics(capabilities)
        return current
    
    def classify_hardware(self, capabilities: HardwareCapabilities) -> ResourceClass:
        """Generic hardware classification"""
        if capabilities.memory_total >= 4:
            return ResourceClass.MEDIUM
        else:
            return ResourceClass.LOW
    
    def get_platform_specific_info(self, capabilities: HardwareCapabilities) -> Dict[str, Any]:
        """Generic platform information"""
        return {
            'platform': capabilities.platform.value,
            'compute_capability': capabilities.compute_capability.value
        }