"""
Template Generation and Basket-based Organization System

Manages pipeline templates for efficient vehicle-to-vehicle pipeline formation.
Provides fast template lookup (<5ms) and organized storage using basket-based clustering.
"""
import time
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
import heapq

from fhdp.core.types import (
    PipelineTemplate, Pipeline, VehicleInfo, ResourceClass, 
    TrainingConfig, MobilityPrediction
)
from fhdp.core.constants import (
    TEMPLATE_CACHE_SIZE, MAX_PIPELINE_LENGTH, MIN_PIPELINE_PARTICIPANTS,
    TEMPLATE_LOOKUP_LATENCY_THRESHOLD, TEMPLATE_GENERATION_INTERVAL,
    MAX_TEMPLATE_MEMORY
)

@dataclass
class TemplateBasket:
    """Basket for organizing similar templates"""
    basket_id: str
    resource_signature: str  # Hash of resource requirements
    templates: List[PipelineTemplate] = field(default_factory=list)
    avg_success_rate: float = 0.0
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    
class TemplateGenerator:
    """Generates new pipeline templates based on observed patterns"""
    
    def __init__(self):
        self.observed_pipelines = []
        self.success_patterns = defaultdict(float)
        self.resource_usage_stats = defaultdict(list)
        
    def generate_template_from_pipeline(self, pipeline: Pipeline, success_rate: float) -> PipelineTemplate:
        """Generate template from successful pipeline execution"""
        template_id = self._generate_template_id(pipeline)
        
        # Extract resource requirements from pipeline
        resource_requirements = self._extract_resource_pattern(pipeline)
        
        # Estimate communication pattern
        comm_pattern = self._infer_communication_pattern(pipeline)
        
        # Create training config
        training_config = self._create_training_config(pipeline)
        
        template = PipelineTemplate(
            template_id=template_id,
            resource_requirements=resource_requirements,
            expected_duration=pipeline.expected_completion - pipeline.start_time,
            communication_pattern=comm_pattern,
            training_config=training_config,
            model_fragment_size=self._estimate_fragment_size(pipeline)
        )
        
        # Update generation statistics
        self.observed_pipelines.append((template, success_rate))
        self.success_patterns[template_id] = success_rate
        
        return template
    
    def _generate_template_id(self, pipeline: Pipeline) -> str:
        """Generate unique template ID"""
        content = f"{len(pipeline.vehicles)}_{len(pipeline.stages)}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _extract_resource_pattern(self, pipeline: Pipeline) -> List[ResourceClass]:
        """Extract resource requirement pattern from pipeline"""
        # TODO: In a real implementation, this should use actual vehicle resource data
        # For now, use a more sophisticated pattern based on pipeline length and position
        resource_pattern = []
        n_vehicles = len(pipeline.vehicles)
        
        for i in range(n_vehicles):
            if i == 0 or i == n_vehicles - 1:
                # First and last vehicles typically need higher resources
                resource_pattern.append(ResourceClass.HIGH)
            elif i == 1 or i == n_vehicles - 2:
                # Second and second-to-last vehicles need medium-high resources
                resource_pattern.append(ResourceClass.MEDIUM)
            else:
                # Middle vehicles can vary based on position
                # Front middle vehicles handle more intermediate results
                if i < n_vehicles / 2:
                    resource_pattern.append(ResourceClass.MEDIUM)
                else:
                    # Rear middle vehicles can use lower resources
                    resource_pattern.append(ResourceClass.LOW)
        
        return resource_pattern
    
    def _infer_communication_pattern(self, pipeline: Pipeline) -> List[Tuple[int, int]]:
        """Infer communication pattern from pipeline structure"""
        pattern = []
        n_stages = len(pipeline.stages)
        
        # Linear pipeline communication
        for i in range(n_stages - 1):
            pattern.append((i, i + 1))
        
        # Add some cross-connections for robustness
        if n_stages > 2:
            pattern.append((0, n_stages - 1))  # Direct connection
            
        return pattern
    
    def _create_training_config(self, pipeline: Pipeline) -> TrainingConfig:
        """Create training configuration based on pipeline"""
        # Adjust epochs based on pipeline length
        epochs = max(1, 2 // len(pipeline.vehicles))
        
        return TrainingConfig(
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
            local_data_size=1000,  # Would be based on actual data
            communication_budget=1024 * 1024
        )
    
    def _estimate_fragment_size(self, pipeline: Pipeline) -> int:
        """Estimate model fragment size for pipeline"""
        # Assume total model size of 100MB divided among vehicles
        total_model_size = 100 * 1024 * 1024  # 100MB
        return total_model_size // len(pipeline.vehicles)
    
    def generate_synthetic_templates(self, num_templates: int = 50) -> List[PipelineTemplate]:
        """Generate synthetic templates for diverse scenarios"""
        templates = []
        
        for i in range(num_templates):
            # Vary pipeline length
            length = np.random.randint(MIN_PIPELINE_PARTICIPANTS, MAX_PIPELINE_LENGTH + 1)
            
            # Generate resource requirements
            resource_requirements = []
            for j in range(length):
                if j == 0 or j == length - 1:
                    # First and last vehicles need higher resources
                    resource_class = np.random.choice([ResourceClass.HIGH, ResourceClass.MEDIUM], p=[0.7, 0.3])
                else:
                    # Middle vehicles have varied requirements
                    resource_class = np.random.choice([ResourceClass.HIGH, ResourceClass.MEDIUM, ResourceClass.LOW], 
                                                    p=[0.3, 0.5, 0.2])
                resource_requirements.append(resource_class)
            
            # Generate communication pattern
            comm_pattern = []
            for j in range(length - 1):
                comm_pattern.append((j, j + 1))
            
            # Add some skip connections
            if length > 3:
                for _ in range(np.random.randint(1, min(3, length - 2))):
                    src, dst = np.random.choice(length, 2, replace=False)
                    if abs(dst - src) > 1:  # Skip connection
                        comm_pattern.append((min(src, dst), max(src, dst)))
            
            # Create training config
            epochs = np.random.randint(1, 3)
            config = TrainingConfig(
                epochs=epochs,
                batch_size=np.random.choice([16, 32, 64]),
                learning_rate=np.random.uniform(0.0001, 0.01),
                local_data_size=np.random.randint(500, 2000),
                communication_budget=np.random.randint(512*1024, 2*1024*1024)
            )
            
            template = PipelineTemplate(
                template_id=f"synth_{i:03d}_{hashlib.md5(str(resource_requirements).encode()).hexdigest()[:8]}",
                resource_requirements=resource_requirements,
                expected_duration=np.random.uniform(5.0, 30.0),
                communication_pattern=comm_pattern,
                training_config=config,
                model_fragment_size=np.random.randint(10*1024*1024, 50*1024*1024)
            )
            
            templates.append(template)
        
        return templates

class TemplateMatcher:
    """Fast template matching engine with <5ms latency guarantee"""
    
    def __init__(self):
        self.baskets: Dict[str, TemplateBasket] = {}
        self.resource_index = defaultdict(set)  # resource_signature -> basket_ids
        self.length_index = defaultdict(set)  # pipeline_length -> basket_ids
        self.template_id_to_basket_id = {}  # template_id -> basket_id for fast lookup
        self.success_cache = {}  # LRU cache for successful matches
        self.cache_size = 1000
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_lookups = 0
        
    def _create_resource_signature(self, resource_requirements: List[ResourceClass]) -> str:
        """Create hash signature for resource requirements"""
        return hashlib.md5(''.join([r.value for r in resource_requirements]).encode()).hexdigest()
    
    def add_template(self, template: PipelineTemplate):
        """Add template to appropriate basket"""
        signature = self._create_resource_signature(template.resource_requirements)
        basket_id = f"{signature}_{len(template.resource_requirements)}"
        
        if basket_id not in self.baskets:
            # Create new basket
            basket = TemplateBasket(
                basket_id=basket_id,
                resource_signature=signature
            )
            self.baskets[basket_id] = basket
            
            # Update indexes
            self.resource_index[signature].add(basket_id)
            self.length_index[len(template.resource_requirements)].add(basket_id)
        
        # Add template to basket
        self.baskets[basket_id].templates.append(template)
        
        # Update template to basket index
        self.template_id_to_basket_id[template.template_id] = basket_id
        
        # Sort templates by expected duration (faster templates first)
        self.baskets[basket_id].templates.sort(key=lambda t: t.expected_duration)
    
    def find_best_template(self, available_vehicles: List[VehicleInfo], 
                          max_candidates: int = 5) -> List[Tuple[PipelineTemplate, float]]:
        """Find best matching templates within 5ms"""
        start_time = time.time()
        
        # Extract vehicle resources
        vehicle_resources = []
        for vehicle in available_vehicles:
            cpu_usage = 1.0 - vehicle.resources.get('cpu', 0.5)
            memory_usage = 1.0 - vehicle.resources.get('memory', 0.5)
            battery = vehicle.resources.get('battery', 0.7)
            
            if cpu_usage > 0.8 and memory_usage > 0.8 and battery > 0.7:
                vehicle_resources.append(ResourceClass.HIGH)
            elif cpu_usage > 0.5 and memory_usage > 0.5 and battery > 0.4:
                vehicle_resources.append(ResourceClass.MEDIUM)
            else:
                vehicle_resources.append(ResourceClass.LOW)
        
        candidates = []
        
        # Check cache first
        # Make cache key order-independent by counting resource classes
        resource_counts = {}
        for r in vehicle_resources:
            resource_counts[r.value] = resource_counts.get(r.value, 0) + 1
        # Create cache key from sorted resource counts
        cache_key = ''.join([f"{k}:{v}," for k, v in sorted(resource_counts.items())])
        
        # Track cache statistics
        self.cache_lookups += 1
        
        if cache_key in self.success_cache:
            self.cache_hits += 1
            cached_result = self.success_cache[cache_key]
            candidates.extend(cached_result)
            
            # Check time constraint
            if time.time() - start_time > TEMPLATE_LOOKUP_LATENCY_THRESHOLD:
                return candidates[:max_candidates]
        
        # Search for matching templates
        n_vehicles = len(available_vehicles)
        
        # First try exact length matches
        matching_baskets = self.length_index.get(n_vehicles, set())
        
        for basket_id in matching_baskets:
            basket = self.baskets[basket_id]
            
            for template in basket.templates:
                score = self._calculate_match_score(template, vehicle_resources)
                if score > 0.3:  # Minimum match threshold
                    candidates.append((template, score))
                
                # Check time constraint
                if time.time() - start_time > TEMPLATE_LOOKUP_LATENCY_THRESHOLD:
                    break
            
            if time.time() - start_time > TEMPLATE_LOOKUP_LATENCY_THRESHOLD:
                break
        
        # If no exact matches, try flexible length matching
        if not candidates and n_vehicles > MIN_PIPELINE_PARTICIPANTS:
            for length in range(MIN_PIPELINE_PARTICIPANTS, min(n_vehicles, MAX_PIPELINE_LENGTH) + 1):
                if length == n_vehicles:
                    continue
                    
                matching_baskets = self.length_index.get(length, set())
                
                for basket_id in matching_baskets:
                    basket = self.baskets[basket_id]
                    
                    for template in basket.templates:
                        score = self._calculate_match_score(template, vehicle_resources, flexible=True)
                        if score > 0.2:
                            candidates.append((template, score * 0.8))  # Penalty for length mismatch
                    
                    if time.time() - start_time > TEMPLATE_LOOKUP_LATENCY_THRESHOLD:
                        break
                
                if time.time() - start_time > TEMPLATE_LOOKUP_LATENCY_THRESHOLD:
                    break
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        result = candidates[:max_candidates]
        
        # Update cache
        if result:
            if len(self.success_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.success_cache))
                del self.success_cache[oldest_key]
            self.success_cache[cache_key] = result
        
        return result
    
    def _calculate_match_score(self, template: PipelineTemplate, 
                             available_resources: List[ResourceClass],
                             flexible: bool = False) -> float:
        """Calculate match score between template and available resources"""
        template_resources = template.resource_requirements
        
        if not flexible and len(template_resources) != len(available_resources):
            return 0.0
        
        if flexible and len(template_resources) > len(available_resources):
            return 0.0
        
        # Calculate resource matching score
        score = 0.0
        n_matched = min(len(template_resources), len(available_resources))
        
        for i in range(n_matched):
            required = template_resources[i]
            available = available_resources[i]
            
            # Resource level matching
            if required == ResourceClass.LOW:
                score += 1.0 if available in [ResourceClass.LOW, ResourceClass.MEDIUM, ResourceClass.HIGH] else 0.5
            elif required == ResourceClass.MEDIUM:
                score += 1.0 if available in [ResourceClass.MEDIUM, ResourceClass.HIGH] else 0.3
            elif required == ResourceClass.HIGH:
                score += 1.0 if available == ResourceClass.HIGH else 0.1
        
        score /= len(template_resources)
        
        # Apply basket success rate bonus
        signature = self._create_resource_signature(template.resource_requirements)
        # Find the basket that contains this template
        for basket_id in self.resource_index[signature]:
            basket = self.baskets[basket_id]
            if template in basket.templates:
                score *= (1.0 + basket.avg_success_rate * 0.2)
                break  # Apply bonus only once from the correct basket
        
        return score
    
    def update_basket_statistics(self, template_id: str, success: bool):
        """Update basket statistics after template usage"""
        # Use index for fast lookup
        if template_id in self.template_id_to_basket_id:
            basket_id = self.template_id_to_basket_id[template_id]
            basket = self.baskets[basket_id]
            
            # Find the template in the basket
            for template in basket.templates:
                if template.template_id == template_id:
                    basket.usage_count += 1
                    basket.last_used = time.time()
                    
                    # Update success rate with exponential moving average
                    if basket.avg_success_rate == 0.0:
                        basket.avg_success_rate = 1.0 if success else 0.0
                    else:
                        alpha = 0.1  # Learning rate
                        basket.avg_success_rate = (1 - alpha) * basket.avg_success_rate + alpha * (1.0 if success else 0.0)
                    break

class TemplateManager:
    """Main template management service"""
    
    def __init__(self):
        self.generator = TemplateGenerator()
        self.matcher = TemplateMatcher()
        self.last_generation_time = 0.0
        self.total_memory_usage = 0
        
        # Initialize with templates from registry
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize template system with templates from registry"""
        from fhdp.core.pipeline_model import PIPELINE_TEMPLATE_REGISTRY
        
        # Add templates from the registry (these have model_partition info)
        for template in PIPELINE_TEMPLATE_REGISTRY.values():
            self.matcher.add_template(template)
        
        # Also add some synthetic templates for broader coverage
        synthetic_templates = self.generator.generate_synthetic_templates(50)
        for template in synthetic_templates:
            self.matcher.add_template(template)
    
    def _classify_vehicle_resource(self, vehicle: VehicleInfo) -> ResourceClass:
        """Classify vehicle resource level based on capabilities"""
        resources = vehicle.resources
        
        # Get memory in GB
        memory_gb = resources.get("memory_gb", resources.get("memory", 0))
        if isinstance(memory_gb, (int, float)):
            memory_gb = float(memory_gb)
        else:
            memory_gb = 0.0
        
        # Get compute capability (GPU type, FLOPS, etc.)
        gpu_type = resources.get("gpu_type", "").lower()
        compute_score = resources.get("compute_score", 0.5)
        
        # Classification logic for Jetson devices:
        # HIGH: Jetson AGX Orin (64GB), Orin NX (16GB), AGX Xavier (32GB)
        # MEDIUM: Orin Nano (8GB), AGX Xavier (16GB), Xavier NX (8GB)
        # LOW: Jetson Nano (4GB), older devices
        
        # Check for specific Jetson models
        if "orin" in gpu_type:
            # Orin devices
            if memory_gb >= 32 or "agx" in gpu_type:
                return ResourceClass.HIGH  # AGX Orin 64GB
            elif memory_gb >= 16:
                return ResourceClass.HIGH  # Orin NX 16GB
            else:
                return ResourceClass.MEDIUM  # Orin Nano 8GB
        elif "xavier" in gpu_type:
            # Xavier devices
            if "agx" in gpu_type and memory_gb >= 32:
                return ResourceClass.HIGH  # AGX Xavier 32GB
            elif memory_gb >= 16:
                return ResourceClass.MEDIUM  # AGX Xavier 16GB, Xavier NX
            else:
                return ResourceClass.MEDIUM  # Xavier NX 8GB
        elif "nano" in gpu_type:
            return ResourceClass.LOW  # Jetson Nano 4GB
        else:
            # Generic classification based on memory and compute score
            if memory_gb >= 32 or compute_score >= 0.8:
                return ResourceClass.HIGH
            elif memory_gb >= 8 or compute_score >= 0.5:
                return ResourceClass.MEDIUM
            else:
                return ResourceClass.LOW
    
    def find_template_for_vehicles(self, vehicles: List[VehicleInfo]) -> Optional[PipelineTemplate]:
        """Find best template for given vehicles based on their resources"""
        from fhdp.core.pipeline_model import PIPELINE_TEMPLATE_REGISTRY
        
        if not vehicles:
            return None
        
        # Classify each vehicle's resource level
        vehicle_resources = [self._classify_vehicle_resource(v) for v in vehicles]
        
        # First, try to find exact match in registry
        best_template = None
        best_score = -1
        
        for template in PIPELINE_TEMPLATE_REGISTRY.values():
            if len(template.resource_requirements) != len(vehicles):
                continue
            
            # Calculate match score
            score = 0
            for i, (req, actual) in enumerate(zip(template.resource_requirements, vehicle_resources)):
                if req == actual:
                    score += 2  # Exact match
                elif (req == ResourceClass.HIGH and actual == ResourceClass.MEDIUM) or \
                     (req == ResourceClass.MEDIUM and actual == ResourceClass.HIGH):
                    score += 1  # Close match
                elif (req == ResourceClass.MEDIUM and actual == ResourceClass.LOW) or \
                     (req == ResourceClass.LOW and actual == ResourceClass.MEDIUM):
                    score += 0.5  # Acceptable match
            
            # Prefer templates with model_partition
            if template.model_partition:
                score += 3
                
                # Add score based on resource estimates matching
                resource_estimates = template.model_partition.get("resource_estimates", {})
                for i, vehicle in enumerate(vehicles):
                    stage_key = f"stage{i}"
                    if stage_key in resource_estimates:
                        est = resource_estimates[stage_key]
                        vehicle_mem = vehicle.resources.get("memory_gb", 0)
                        vehicle_score = vehicle.resources.get("compute_score", 0)
                        
                        # Check if memory is sufficient
                        if vehicle_mem >= est.get("memory_gb", 0):
                            score += 1.5
                        elif vehicle_mem >= est.get("memory_gb", 0) * 0.8:
                            score += 1.0
                        
                        # Check if compute score is sufficient
                        if vehicle_score >= est.get("compute_score", 0):
                            score += 1.5
                        elif vehicle_score >= est.get("compute_score", 0) * 0.8:
                            score += 1.0
            
            if score > best_score:
                best_score = score
                best_template = template
        
        if best_template:
            return best_template
        
        # Fallback to matcher-based search
        candidates = self.matcher.find_best_template(vehicles)
        if candidates:
            return candidates[0][0]
        
        return None
    
    def register_successful_pipeline(self, pipeline: Pipeline, success: bool, duration: float):
        """Register pipeline execution for template learning"""
        # Generate new template if successful
        if success:
            template = self.generator.generate_template_from_pipeline(pipeline, 1.0)
            self.matcher.add_template(template)
        
        # Update statistics
        if hasattr(pipeline, 'template_id') and pipeline.template_id:
            self.matcher.update_basket_statistics(pipeline.template_id, success)
        
        # Periodic template generation
        current_time = time.time()
        if current_time - self.last_generation_time > TEMPLATE_GENERATION_INTERVAL:
            self._generate_new_templates()
            self.last_generation_time = current_time
    
    def _generate_new_templates(self):
        """Generate new templates based on recent patterns"""
        # Generate templates for underrepresented patterns
        new_templates = self.generator.generate_synthetic_templates(20)
        
        for template in new_templates:
            # Only add templates for patterns not well represented
            # Create realistic vehicle resources based on template requirements
            vehicles_with_resources = []
            for i, resource_req in enumerate(template.resource_requirements):
                # Assign resources based on required resource class
                if resource_req == ResourceClass.HIGH:
                    resources = {'cpu': 0.9, 'memory': 0.85, 'battery': 0.8}
                elif resource_req == ResourceClass.MEDIUM:
                    resources = {'cpu': 0.7, 'memory': 0.65, 'battery': 0.6}
                else:  # LOW
                    resources = {'cpu': 0.5, 'memory': 0.45, 'battery': 0.4}
                
                vehicle = VehicleInfo(str(i), (0, 0), 0, 0, resources)
                vehicles_with_resources.append(vehicle)
            
            # Check if template is underrepresented
            candidates = self.matcher.find_best_template(
                vehicles_with_resources,
                max_candidates=1
            )
            
            if not candidates or candidates[0][1] < 0.5:
                self.matcher.add_template(template)
    
    def get_template_statistics(self) -> Dict[str, any]:
        """Get template system statistics"""
        total_templates = sum(len(basket.templates) for basket in self.matcher.baskets.values())
        avg_success_rate = np.mean([basket.avg_success_rate for basket in self.matcher.baskets.values()])
        
        # Calculate cache hit rate
        cache_lookups = self.matcher.cache_lookups
        cache_hits = self.matcher.cache_hits
        cache_hit_rate = cache_hits / max(1, cache_lookups)
        
        return {
            "total_baskets": len(self.matcher.baskets),
            "total_templates": total_templates,
            "avg_success_rate": avg_success_rate,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": cache_hits,
            "cache_lookups": cache_lookups,
            "memory_usage": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of template system"""
        usage = 0
        for basket in self.matcher.baskets.values():
            for template in basket.templates:
                usage += len(str(template.resource_requirements)) * 8
                usage += len(template.template_id) * 8
                usage += len(template.communication_pattern) * 16
                usage += 1024  # Estimated overhead per template
        return usage