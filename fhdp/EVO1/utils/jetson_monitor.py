#!/usr/bin/env python3
"""
Jetson Device Resource Monitoring Tool

This tool provides real-time monitoring of Jetson device resources:
- Memory usage (GPU/CPU)
- Temperature
- Power consumption
- Process information
- Performance metrics

Designed for EVO-1 Stage 1 deployment on Jetson Orin/Nano devices.
"""

import time
import psutil
import torch
import os
from typing import Dict, Any, List
from dataclasses import dataclass
import threading

@dataclass
class JetsonMonitorConfig:
    """Configuration for Jetson Monitor"""
    update_interval: float = 1.0  # Update interval in seconds
    memory_threshold: float = 80.0  # Memory usage threshold in percent
    temperature_threshold: float = 85.0  # Temperature threshold in Celsius
    log_file: str = "jetson_monitor.log"
    log_enabled: bool = True

class JetsonMonitor:
    """Jetson Device Resource Monitor"""
    
    def __init__(self, config: JetsonMonitorConfig = None):
        self.config = config or JetsonMonitorConfig()
        self.running = False
        self.metrics_history = []
        self.monitor_thread = None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information"""
        metrics = {}
        
        # CPU memory usage
        cpu_mem = psutil.virtual_memory()
        metrics.update({
            'cpu_total_gb': round(cpu_mem.total / (1024**3), 2),
            'cpu_used_gb': round(cpu_mem.used / (1024**3), 2),
            'cpu_usage_percent': cpu_mem.percent
        })
        
        # GPU memory usage (if available)
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_mem_cached = torch.cuda.memory_reserved() / (1024**3)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            metrics.update({
                'gpu_total_gb': round(gpu_mem_total, 2),
                'gpu_used_gb': round(gpu_mem_allocated, 2),
                'gpu_cached_gb': round(gpu_mem_cached, 2),
                'gpu_usage_percent': round((gpu_mem_allocated / gpu_mem_total) * 100, 1)
            })
        
        return metrics
    
    def get_temperature(self) -> Dict[str, float]:
        """Get current device temperature"""
        metrics = {}
        
        # Try to get temperature from sysfs (Jetson-specific)
        try:
            # Check for Jetson thermal zones
            thermal_zones = [f for f in os.listdir('/sys/class/thermal') if f.startswith('thermal_zone')]
            
            for zone in thermal_zones:
                zone_path = f'/sys/class/thermal/{zone}'
                try:
                    with open(f'{zone_path}/type', 'r') as f:
                        zone_type = f.read().strip()
                    
                    with open(f'{zone_path}/temp', 'r') as f:
                        temp_milli = int(f.read().strip())
                        temp_celsius = temp_milli / 1000.0
                    
                    metrics[zone_type] = round(temp_celsius, 1)
                except Exception:
                    continue
        except Exception:
            # Fallback to psutil if sysfs access fails
            pass
        
        return metrics
    
    def get_power_usage(self) -> Dict[str, float]:
        """Get current power consumption"""
        metrics = {}
        
        # Try to get power from sysfs (Jetson-specific)
        try:
            # Check for Jetson power rails
            power_rails = [f for f in os.listdir('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0') \
                          if f.startswith('in_power') and f.endswith('_input')]
            
            for rail in power_rails:
                rail_path = f'/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/{rail}'
                try:
                    with open(rail_path, 'r') as f:
                        power_microwatts = int(f.read().strip())
                        power_watts = power_microwatts / 1000000.0
                    
                    rail_name = rail.replace('in_power', '').replace('_input', '').strip('_')
                    metrics[f'power_{rail_name}'] = round(power_watts, 3)
                except Exception:
                    continue
        except Exception:
            # Fallback if power monitoring is not available
            pass
        
        return metrics
    
    def get_process_info(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get information about top processes by CPU usage"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 0.0:
                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage and get top N
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        return processes[:top_n]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all available metrics"""
        timestamp = time.time()
        
        metrics = {
            'timestamp': timestamp,
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
            'memory': self.get_memory_usage(),
            'temperature': self.get_temperature(),
            'power': self.get_power_usage(),
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'load_avg': psutil.getloadavg(),
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to file"""
        if not self.config.log_enabled:
            return
        
        try:
            with open(self.config.log_file, 'a') as f:
                log_line = f"{metrics['datetime']} "
                log_line += f"CPU:{metrics['cpu_usage']:.1f}% "
                log_line += f"MEM:{metrics['memory'].get('cpu_usage_percent', 0):.1f}% "
                
                if 'gpu_usage_percent' in metrics['memory']:
                    log_line += f"GPU_MEM:{metrics['memory']['gpu_usage_percent']:.1f}% "
                
                if metrics['temperature']:
                    temp_str = ",".join([f"{k}:{v:.1f}C" for k, v in metrics['temperature'].items()])
                    log_line += f"TEMP:{temp_str} "
                
                if metrics['power']:
                    power_total = sum(metrics['power'].values())
                    log_line += f"POWER:{power_total:.2f}W "
                
                f.write(log_line.strip() + '\n')
        except Exception as e:
            print(f"Error logging metrics: {e}")
    
    def monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            metrics = self.get_all_metrics()
            self.metrics_history.append(metrics)
            self.log_metrics(metrics)
            
            # Keep history size manageable
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            time.sleep(self.config.update_interval)
    
    def start(self) -> None:
        """Start the monitoring thread"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("🚀 Jetson Monitor started")
    
    def stop(self) -> None:
        """Stop the monitoring thread"""
        if self.running:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join()
            print("✅ Jetson Monitor stopped")
    
    def print_live_stats(self, duration: float = 10.0) -> None:
        """Print live statistics for a specified duration"""
        print("📊 Jetson Live Statistics")
        print("=" * 60)
        print(f"Updating every {self.config.update_interval} seconds, running for {duration} seconds")
        print("=" * 60)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            metrics = self.get_all_metrics()
            
            # Clear line and print
            print(f"\rCPU: {metrics['cpu_usage']:.1f}% | "
                  f"MEM: {metrics['memory']['cpu_usage_percent']:.1f}% | "
                  f"GPU: {metrics['memory'].get('gpu_usage_percent', 0):.1f}% | "
                  f"TEMP: {max(metrics['temperature'].values()) if metrics['temperature'] else 0:.1f}°C | "
                  f"POWER: {sum(metrics['power'].values()):.2f}W", end="", flush=True)
            
            time.sleep(self.config.update_interval)
        
        print("\n" + "=" * 60)
        print("✅ Live statistics complete")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jetson Device Resource Monitor")
    parser.add_argument("-d", "--duration", type=float, default=10.0, 
                        help="Duration for live monitoring (seconds)")
    parser.add_argument("-i", "--interval", type=float, default=1.0, 
                        help="Update interval (seconds)")
    parser.add_argument("--log", action="store_true", 
                        help="Enable logging to file")
    
    args = parser.parse_args()
    
    # Create monitor instance
    config = JetsonMonitorConfig(
        update_interval=args.interval,
        log_enabled=args.log
    )
    
    monitor = JetsonMonitor(config)
    
    # Run live stats
    monitor.print_live_stats(duration=args.duration)
    
    # Print detailed metrics
    print("\n📋 Detailed Metrics:")
    print("=" * 60)
    metrics = monitor.get_all_metrics()
    
    print("Memory Usage:")
    for k, v in metrics['memory'].items():
        print(f"  {k}: {v}")
    
    print("\nTemperature:")
    for k, v in metrics['temperature'].items():
        print(f"  {k}: {v}°C")
    
    print("\nPower Consumption:")
    for k, v in metrics['power'].items():
        print(f"  {k}: {v}W")
    
    print("\nProcess Information (Top 5 by CPU):")
    processes = monitor.get_process_info()
    for proc in processes:
        print(f"  PID:{proc['pid']} {proc['name']}: CPU {proc['cpu_percent']:.1f}%, MEM {proc['memory_percent']:.1f}%")

if __name__ == "__main__":
    main()
