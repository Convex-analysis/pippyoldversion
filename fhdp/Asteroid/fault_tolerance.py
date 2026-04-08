"""Fault tolerance manager for Asteroid"""

from typing import Dict, List, Tuple, Optional
import time
import threading
import torch
import torch.nn as nn

class FaultToleranceManager:
    """Fault tolerance manager for handling device failures"""
    
    def __init__(self, devices: List[str], heartbeat_interval: float = 5.0, timeout: float = 15.0):
        """
        Initialize the fault tolerance manager
        
        Args:
            devices: List of device identifiers
            heartbeat_interval: Heartbeat interval in seconds
            timeout: Timeout in seconds
        """
        self.devices = devices
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self.heartbeats = {device: time.time() for device in devices}
        self.failed_devices = set()
        self.lock = threading.Lock()
        self.heartbeat_thread = None
        self.running = False
    
    def start(self):
        """
        Start the heartbeat monitoring thread
        """
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._monitor_heartbeats, daemon=True)
        self.heartbeat_thread.start()
    
    def stop(self):
        """
        Stop the heartbeat monitoring thread
        """
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join()
    
    def update_heartbeat(self, device: str):
        """
        Update heartbeat for a device
        
        Args:
            device: Device identifier
        """
        with self.lock:
            if device in self.heartbeats:
                self.heartbeats[device] = time.time()
                # Remove from failed devices if it was marked as failed
                if device in self.failed_devices:
                    self.failed_devices.remove(device)
    
    def is_device_alive(self, device: str) -> bool:
        """
        Check if a device is alive
        
        Args:
            device: Device identifier
            
        Returns:
            True if device is alive
        """
        with self.lock:
            if device in self.failed_devices:
                return False
            if device not in self.heartbeats:
                return False
            return time.time() - self.heartbeats[device] < self.timeout
    
    def get_failed_devices(self) -> List[str]:
        """
        Get list of failed devices
        
        Returns:
            List of failed devices
        """
        with self.lock:
            return list(self.failed_devices)
    
    def _monitor_heartbeats(self):
        """
        Monitor heartbeats and detect failures
        """
        while self.running:
            time.sleep(self.heartbeat_interval)
            with self.lock:
                current_time = time.time()
                for device, last_heartbeat in self.heartbeats.items():
                    if current_time - last_heartbeat > self.timeout:
                        if device not in self.failed_devices:
                            self.failed_devices.add(device)
                            print(f"[Fault Tolerance] Device {device} marked as failed")
    
    def replicate_model(self, model: nn.Module, stage_devices: Dict[int, List[str]]) -> Dict[str, nn.Module]:
        """
        Replicate model across devices for fault tolerance
        
        Args:
            model: PyTorch model
            stage_devices: Dictionary mapping stage index to list of devices
            
        Returns:
            Dictionary mapping device to model replica
        """
        replicas = {}
        for stage, devices in stage_devices.items():
            # For single-device stages, replicate to next stage's device
            if len(devices) == 1:
                device = devices[0]
                # Create replica
                replica = type(model)()
                replica.load_state_dict(model.state_dict())
                replicas[device] = replica
                
                # Find next stage's devices
                next_stage = stage + 1
                if next_stage in stage_devices:
                    next_devices = stage_devices[next_stage]
                    for next_device in next_devices:
                        # Create backup replica
                        backup_replica = type(model)()
                        backup_replica.load_state_dict(model.state_dict())
                        replicas[next_device] = backup_replica
            else:
                # For multi-device stages, each device has its own replica
                for device in devices:
                    replica = type(model)()
                    replica.load_state_dict(model.state_dict())
                    replicas[device] = replica
        
        return replicas
    
    def recover_from_failure(self, failed_device: str, stage_devices: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """
        Recover from device failure by reconfiguring the pipeline
        
        Args:
            failed_device: Failed device identifier
            stage_devices: Dictionary mapping stage index to list of devices
            
        Returns:
            Updated stage devices mapping
        """
        # This is a simplified implementation
        # In practice, you would perform lightweight layer migration
        updated_stage_devices = stage_devices.copy()
        
        # Remove failed device from all stages
        for stage, devices in updated_stage_devices.items():
            if failed_device in devices:
                devices.remove(failed_device)
                # If stage has no devices left, redistribute layers
                if not devices:
                    # Find adjacent stages
                    prev_stage = stage - 1
                    next_stage = stage + 1
                    
                    if prev_stage >= 0:
                        # Merge with previous stage
                        updated_stage_devices[prev_stage].extend(devices)
                        del updated_stage_devices[stage]
                    elif next_stage in updated_stage_devices:
                        # Merge with next stage
                        updated_stage_devices[next_stage].extend(devices)
                        del updated_stage_devices[stage]
        
        return updated_stage_devices
    
    def migrate_layers(self, source_device: str, target_device: str, layers: List[str]) -> bool:
        """
        Migrate layers from source to target device
        
        Args:
            source_device: Source device identifier
            target_device: Target device identifier
            layers: List of layers to migrate
            
        Returns:
            True if migration was successful
        """
        # This is a simplified implementation
        # In practice, you would transfer model weights and update the pipeline
        print(f"[Fault Tolerance] Migrating layers {layers} from {source_device} to {target_device}")
        # Simulate migration time
        time.sleep(0.5)
        return True
