"""Neuron to device mapping algorithm for EdgePipe"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import random
from .super_neuron import SuperNeuronNetwork

class NeuronDeviceMapping:
    """Neuron to device mapping using genetic algorithm"""
    
    def __init__(self, super_neuron_network: SuperNeuronNetwork, device_network: np.ndarray):
        """
        Initialize the neuron to device mapping
        
        Args:
            super_neuron_network: Network of super neurons
            device_network: Device network adjacency matrix (PRR values)
        """
        self.super_neuron_network = super_neuron_network
        self.device_network = device_network
        self.num_devices = device_network.shape[0]
        self.num_super_neurons = super_neuron_network.get_total_super_neurons()
        self.sn_network = self._build_super_neuron_network()
        self.best_mapping = None
        self.best_score = -1
    
    def _build_super_neuron_network(self) -> np.ndarray:
        """
        Build super neuron network adjacency matrix based on training communication needs
        
        Returns:
            Super neuron network adjacency matrix
        """
        super_neurons = self.super_neuron_network.get_super_neurons()
        sn_network = np.zeros((self.num_super_neurons, self.num_super_neurons))
        
        # Build communication matrix based on layer dependencies
        for i, sn1 in enumerate(super_neurons):
            for j, sn2 in enumerate(super_neurons):
                if i == j:
                    continue
                
                # Check if sn1 and sn2 are adjacent in the DNN
                sn1_layers = sn1.get_layers()
                sn2_layers = sn2.get_layers()
                
                # If sn1's highest layer is just before sn2's lowest layer, they need to communicate
                if max(sn1_layers) + 1 == min(sn2_layers):
                    sn_network[i][j] = 1.0
                elif max(sn2_layers) + 1 == min(sn1_layers):
                    sn_network[i][j] = 1.0
        
        return sn_network
    
    def _calculate_score(self, mapping: List[int]) -> float:
        """
        Calculate score for a mapping
        
        Args:
            mapping: List where index is super neuron ID, value is device ID
            
        Returns:
            Score based on Hadamard product of the two networks
        """
        score = 0.0
        
        for i in range(self.num_super_neurons):
            for j in range(self.num_super_neurons):
                if self.sn_network[i][j] > 0:
                    device_i = mapping[i]
                    device_j = mapping[j]
                    # Use PRR value as communication quality
                    score += self.sn_network[i][j] * self.device_network[device_i][device_j]
        
        return score
    
    def _generate_initial_mapping(self) -> List[int]:
        """
        Generate initial mapping
        
        Returns:
            Initial mapping
        """
        # Distribute super neurons evenly across devices
        mapping = []
        for i in range(self.num_super_neurons):
            mapping.append(i % self.num_devices)
        return mapping
    
    def _mutate(self, mapping: List[int]) -> List[int]:
        """
        Mutate a mapping by swapping two elements
        
        Args:
            mapping: Current mapping
            
        Returns:
            Mutated mapping
        """
        new_mapping = mapping.copy()
        # Random swap
        if self.num_super_neurons > 1:
            i, j = random.sample(range(self.num_super_neurons), 2)
            new_mapping[i], new_mapping[j] = new_mapping[j], new_mapping[i]
        return new_mapping
    
    def _swipe(self, mapping: List[int]) -> List[int]:
        """
        Generate a swipe mutation (cycle shift)
        
        Args:
            mapping: Current mapping
            
        Returns:
            Swiped mapping
        """
        new_mapping = mapping.copy()
        # Cycle shift: 2→…→N_devices→1
        for i in range(len(new_mapping)):
            new_mapping[i] = (new_mapping[i] + 1) % self.num_devices
        return new_mapping
    
    def optimize_mapping(self, generations: int = 1000) -> List[int]:
        """
        Optimize mapping using genetic algorithm
        
        Args:
            generations: Number of generations
            
        Returns:
            Best mapping found
        """
        # Generate initial mapping
        current_mapping = self._generate_initial_mapping()
        self.best_mapping = current_mapping.copy()
        self.best_score = self._calculate_score(current_mapping)
        
        for _ in range(generations):
            # Generate two mutations
            mutation1 = self._mutate(current_mapping)
            mutation2 = self._swipe(current_mapping)
            
            # Calculate scores
            score1 = self._calculate_score(mutation1)
            score2 = self._calculate_score(mutation2)
            current_score = self._calculate_score(current_mapping)
            
            # Select best mapping
            if score1 > self.best_score:
                self.best_score = score1
                self.best_mapping = mutation1.copy()
                current_mapping = mutation1
            elif score2 > self.best_score:
                self.best_score = score2
                self.best_mapping = mutation2.copy()
                current_mapping = mutation2
            elif max(score1, score2) > current_score:
                # Choose between the two mutations
                if score1 > score2:
                    current_mapping = mutation1
                else:
                    current_mapping = mutation2
        
        # Apply the best mapping to super neurons
        super_neurons = self.super_neuron_network.get_super_neurons()
        for i, sn in enumerate(super_neurons):
            sn.set_device(self.best_mapping[i])
        
        return self.best_mapping
    
    def get_best_mapping(self) -> Optional[List[int]]:
        """
        Get the best mapping found
        
        Returns:
            Best mapping
        """
        return self.best_mapping
    
    def get_best_score(self) -> float:
        """
        Get the best score found
        
        Returns:
            Best score
        """
        return self.best_score
    
    def get_mapping_info(self) -> Dict:
        """
        Get mapping information
        
        Returns:
            Dict with mapping details
        """
        if self.best_mapping is None:
            return {}
        
        # Count super neurons per device
        device_counts = {}
        for device_id in self.best_mapping:
            if device_id not in device_counts:
                device_counts[device_id] = 0
            device_counts[device_id] += 1
        
        return {
            'best_mapping': self.best_mapping,
            'best_score': self.best_score,
            'super_neurons_per_device': device_counts
        }
