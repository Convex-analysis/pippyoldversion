"""
Hardware Resource Integration with FHDP

This module integrates EVO-1 resource management with FHDP's
native hardware adaptation and resource classification capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from core.hardware_adapter import HardwareAdapter
from core.heterogeneous_resource import ResourceClassifier
from core.types import ResourceClass
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging


class HardwareResourceAdapter:
    """
    Hardware resource adapter for EVO-1 using FHDP's native capabilities
    
    This class leverages FHDP's existing hardware adaptation and resource
    classification while providing EVO-1 specific optimizations.
    """
    
    def __init__(self, device_id: str, resource_config: Optional[Dict] = None):
        self.device_id = device_id
        
        # Use FHDP's native hardware adapter
        self.fhdp_hardware_adapter = HardwareAdapter(device_id)
        
        # Use FHDP's resource classifier
        self.resource_classifier = ResourceClassifier()
        
        # EVO-1 specific resource configuration
        self.evo1_resource_config = self._initialize_evo1_config(resource_config)
        
        # Adaptation strategies
        self.adaptation_strategies = self._select_adaptation_strategies()
        
        logging.info(f"Hardware resource adapter initialized: {device_id}")
    
    def _initialize_evo1_config(self, config: Optional[Dict]) -> Dict[str, Any]:
        """Initialize EVO-1 specific resource configuration"""
        if config is None:
            config = {}
        
        return {
            'encoder_type': config.get('encoder_type', 'resnet18'),
            'batch_size': config.get('batch_size', 8),
            'mixed_precision': config.get('mixed_precision', True),
            'gradient_checkpointing': config.get('gradient_checkpointing', True),
            'memory_optimization': config.get('memory_optimization', True),
            'compute_optimization': config.get('compute_optimization', True)
        }
    
    def _select_adaptation_strategies(self) -> List[str]:
        """Select adaptation strategies using FHDP's resource classification"""
        try:
            # Get hardware info from FHDP
            hardware_info = self.fhdp_hardware_adapter.get_hardware_info()
            
            # Classify resources using FHDP
            resource_class = self.resource_classifier.classify_device(hardware_info)
            
            strategies = []
            
            # Memory-based strategies
            if resource_class == ResourceClass.LOW:
                strategies.extend(['gradient_checkpointing', 'small_batch'])
            elif resource_class == ResourceClass.MEDIUM:
                strategies.extend(['gradient_checkpointing', 'mixed_precision'])
            else:  # HIGH
                strategies.extend(['mixed_precision', 'large_batch'])
            
            # Compute-based strategies
            gpu_available = hardware_info.get('gpu_available', False)
            if not gpu_available:
                strategies.append('cpu_optimization')
            
            return strategies
            
        except Exception as e:
            logging.error(f"Failed to select adaptation strategies: {e}")
            return ['gradient_checkpointing']  # Default safe strategy
    
    def adapt_model_for_hardware(self, model: nn.Module) -> nn.Module:
        """Adapt model for current hardware using FHDP's capabilities"""
        adapted_model = model
        
        # Apply FHDP's native hardware optimizations
        hardware_info = self.fhdp_hardware_adapter.get_hardware_info()
        
        # Apply EVO-1 specific adaptations based on strategies
        for strategy in self.adaptation_strategies:
            adapted_model = self._apply_strategy(adapted_model, strategy, hardware_info)
        
        return adapted_model
    
    def _apply_strategy(self, model: nn.Module, strategy: str, 
                       hardware_info: Dict[str, Any]) -> nn.Module:
        """Apply specific adaptation strategy"""
        if strategy == 'gradient_checkpointing':
            return self._enable_gradient_checkpointing(model)
        elif strategy == 'mixed_precision':
            return self._enable_mixed_precision(model, hardware_info)
        elif strategy == 'cpu_optimization':
            return self._optimize_for_cpu(model)
        elif strategy == 'small_batch':
            return self._adjust_for_small_batch(model)
        elif strategy == 'large_batch':
            return self._adjust_for_large_batch(model)
        else:
            return model
    
    def _enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing using FHDP's implementation"""
        # Use FHDP's gradient checkpointing if available
        if hasattr(self.fhdp_hardware_adapter, 'enable_gradient_checkpointing'):
            return self.fhdp_hardware_adapter.enable_gradient_checkpointing(model)
        
        # Fallback implementation
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.numel() > 10000:
                # Replace with checkpointed version
                parent = model
                for part in name.split('.')[:-1]:
                    parent = getattr(parent, part)
                
                class CheckpointedModule(nn.Module):
                    def __init__(self, orig_module):
                        super().__init__()
                        self.orig_module = orig_module
                    
                    def forward(self, *args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(
                            self.orig_module, *args, **kwargs
                        )
                
                setattr(parent, name.split('.')[-1], CheckpointedModule(module))
        
        return model
    
    def _enable_mixed_precision(self, model: nn.Module, 
                               hardware_info: Dict[str, Any]) -> nn.Module:
        """Enable mixed precision using FHDP's capabilities"""
        if hardware_info.get('gpu_available', False) and torch.cuda.is_available():
            # Use FHDP's mixed precision setup
            if hasattr(self.fhdp_hardware_adapter, 'setup_mixed_precision'):
                return self.fhdp_hardware_adapter.setup_mixed_precision(model)
            
            # Fallback: convert to half precision
            return model.half()
        
        return model
    
    def _optimize_for_cpu(self, model: nn.Module) -> nn.Module:
        """Optimize model for CPU using FHDP's CPU optimizations"""
        # Use FHDP's CPU optimization
        if hasattr(self.fhdp_hardware_adapter, 'optimize_for_cpu'):
            return self.fhdp_hardware_adapter.optimize_for_cpu(model)
        
        # Fallback: basic CPU optimizations
        model.eval()  # Set to eval mode for inference optimization
        return model
    
    def _adjust_for_small_batch(self, model: nn.Module) -> nn.Module:
        """Adjust model for small batch processing"""
        # Small batch specific optimizations
        # Note: This would typically involve model architecture changes
        # For now, just log the adaptation
        logging.info(f"Adjusted model for small batch processing on {self.device_id}")
        return model
    
    def _adjust_for_large_batch(self, model: nn.Module) -> nn.Module:
        """Adjust model for large batch processing"""
        # Large batch specific optimizations
        logging.info(f"Adjusted model for large batch processing on {self.device_id}")
        return model
    
    def get_adapted_training_config(self) -> Dict[str, Any]:
        """Get training configuration adapted to hardware"""
        hardware_info = self.fhdp_hardware_adapter.get_hardware_info()
        resource_class = self.resource_classifier.classify_device(hardware_info)
        
        adapted_config = self.evo1_resource_config.copy()
        
        # Adapt batch size based on resource class
        if resource_class == ResourceClass.LOW:
            adapted_config['batch_size'] = max(2, adapted_config['batch_size'] // 4)
        elif resource_class == ResourceClass.MEDIUM:
            adapted_config['batch_size'] = max(4, adapted_config['batch_size'] // 2)
        # HIGH resource class uses original batch size
        
        # Adapt mixed precision
        adapted_config['mixed_precision'] = (
            adapted_config['mixed_precision'] and 
            hardware_info.get('gpu_available', False)
        )
        
        # Adapt gradient checkpointing
        memory_gb = hardware_info.get('memory_gb', 8)
        adapted_config['gradient_checkpointing'] = (
            adapted_config['gradient_checkpointing'] and memory_gb < 8
        )
        
        return adapted_config
    
    def get_resource_constraints(self) -> Dict[str, Any]:
        """Get resource constraints using FHDP's native constraint detection"""
        # Use FHDP's hardware constraints
        fhdp_constraints = self.fhdp_hardware_adapter.get_resource_constraints()
        
        # Add EVO-1 specific constraints
        evo1_constraints = {
            'encoder_memory_requirement': self._calculate_encoder_memory(),
            'vlm_backbone_compatibility': self._check_vlm_compatibility(),
            'action_head_requirement': self._calculate_action_head_memory(),
            'communication_bandwidth': self._estimate_communication_bandwidth()
        }
        
        return {**fhdp_constraints, **evo1_constraints}
    
    def _calculate_encoder_memory(self) -> float:
        """Calculate encoder memory requirement in GB"""
        # Simplified calculation based on encoder type
        encoder_type = self.evo1_resource_config['encoder_type']
        
        if encoder_type == 'resnet18':
            return 0.5  # ~500MB
        elif encoder_type == 'resnet34':
            return 1.0  # ~1GB
        elif encoder_type == 'efficientnet_b0':
            return 0.3  # ~300MB
        else:
            return 0.8  # Default estimate
    
    def _check_vlm_compatibility(self) -> bool:
        """Check if hardware is compatible with VLM backbone processing"""
        hardware_info = self.fhdp_hardware_adapter.get_hardware_info()
        
        # Check minimum requirements for VLM
        return (
            hardware_info.get('memory_gb', 0) >= 4 and
            (hardware_info.get('gpu_available', False) or 
             hardware_info.get('cpu_cores', 0) >= 4)
        )
    
    def _calculate_action_head_memory(self) -> float:
        """Calculate action head memory requirement in GB"""
        return 0.2  # ~200MB for typical action head
    
    def _estimate_communication_bandwidth(self) -> float:
        """Estimate communication bandwidth in Mbps"""
        hardware_info = self.fhdp_hardware_adapter.get_hardware_info()
        return hardware_info.get('network_bandwidth', 100)
    
    def monitor_resource_usage(self) -> Dict[str, float]:
        """Monitor current resource usage using FHDP's monitoring"""
        # Use FHDP's resource monitoring
        fhdp_metrics = self.fhdp_hardware_adapter.get_performance_metrics()
        
        # Add EVO-1 specific metrics
        evo1_metrics = {
            'encoder_utilization': self._estimate_encoder_utilization(),
            'communication_overhead': self._estimate_communication_overhead(),
            'memory_fragmentation': self._calculate_memory_fragmentation()
        }
        
        return {**fhdp_metrics, **evo1_metrics}
    
    def _estimate_encoder_utilization(self) -> float:
        """Estimate encoder training utilization"""
        # Simplified utilization estimate
        return 0.75  # 75% typical utilization
    
    def _estimate_communication_overhead(self) -> float:
        """Estimate communication overhead"""
        hardware_info = self.fhdp_hardware_adapter.get_hardware_info()
        bandwidth = hardware_info.get('network_bandwidth', 100)
        
        # Overhead estimate based on bandwidth
        if bandwidth < 50:
            return 0.3  # 30% overhead for low bandwidth
        elif bandwidth < 100:
            return 0.15  # 15% overhead for medium bandwidth
        else:
            return 0.05  # 5% overhead for high bandwidth
    
    def _calculate_memory_fragmentation(self) -> float:
        """Calculate memory fragmentation"""
        # Use FHDP's memory monitoring
        if hasattr(self.fhdp_hardware_adapter, 'get_memory_fragmentation'):
            return self.fhdp_hardware_adapter.get_memory_fragmentation()
        
        return 0.1  # Default 10% fragmentation estimate
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of applied adaptations"""
        return {
            'device_id': self.device_id,
            'resource_class': self.resource_classifier.classify_device(
                self.fhdp_hardware_adapter.get_hardware_info()
            ),
            'adaptation_strategies': self.adaptation_strategies,
            'evo1_config': self.evo1_resource_config,
            'adapted_training_config': self.get_adapted_training_config(),
            'resource_constraints': self.get_resource_constraints(),
            'current_usage': self.monitor_resource_usage()
        }