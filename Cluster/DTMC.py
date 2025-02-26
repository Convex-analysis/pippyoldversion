import numpy as np
import random
from typing import List, Dict, Set, Tuple
import xml.etree.ElementTree as ET

class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        
    def distance_to(self, other: 'Cell') -> int:
        """Calculate distance between two cells in number of cells"""
        return abs(self.x - other.x) + abs(self.y - other.y)  # Manhattan distance

class MobilityPattern:
    def __init__(self, pattern_id: int, transition_matrix: np.ndarray):
        self.pattern_id = pattern_id
        self.transition_matrix = transition_matrix  # P(c_i → c_j|m_k)

class DTMC:
    def __init__(self, area_range: float, unit_distance: float, comm_radius: float):
        self.R = area_range  # Range of current area
        self.rho = unit_distance  # Unit distance (ρ)
        self.R_v = comm_radius  # Communication radius
        
        # Calculate number of cells
        self.num_cells = int((self.R ** 2) / (self.rho ** 2))
        self.cells = self._initialize_cells()
        
        # Mobility patterns
        self.mobility_patterns: Dict[int, MobilityPattern] = {}
        
    def _initialize_cells(self) -> List[Cell]:
        """Initialize grid cells"""
        cells = []
        grid_size = int(np.sqrt(self.num_cells))
        for i in range(grid_size):
            for j in range(grid_size):
                cells.append(Cell(i, j))
        return cells
    
    def add_mobility_pattern(self, pattern_id: int, transition_probs: np.ndarray):
        """Add a new mobility pattern with transition probabilities"""
        if transition_probs.shape != (self.num_cells, self.num_cells):
            raise ValueError("Transition matrix dimensions must match number of cells")
        self.mobility_patterns[pattern_id] = MobilityPattern(pattern_id, transition_probs)
    
    def get_transition_probability(self, from_cell: int, to_cell: int, pattern: int) -> float:
        """Get transition probability P(c_i → c_j|m_k)"""
        if pattern not in self.mobility_patterns:
            raise ValueError(f"Mobility pattern {pattern} not found")
        return self.mobility_patterns[pattern].transition_matrix[from_cell, to_cell]
    
    def predict_next_cell(self, current_cell: int, pattern: int) -> int:
        """Predict next cell based on current cell and mobility pattern"""
        if pattern not in self.mobility_patterns:
            raise ValueError(f"Mobility pattern {pattern} not found")
        
        probs = self.mobility_patterns[pattern].transition_matrix[current_cell]
        return np.random.choice(len(probs), p=probs)
    
    def get_neighbors(self, cell_idx: int) -> Set[int]:
        """Get neighboring cells within Manhattan distance of 1"""
        neighbors = set()
        current_cell = self.cells[cell_idx]
        grid_size = int(np.sqrt(self.num_cells))
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current_cell.x + dx, current_cell.y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbor_idx = nx * grid_size + ny
                neighbors.add(neighbor_idx)
                
        return neighbors
    
    def generate_random_mobility_pattern(self, pattern_id: int, mean: float, stddev: float):
        """
        Generate a random mobility pattern using Gaussian distribution.
        Parameters:
        pattern_id (int): The identifier for the mobility pattern.
        mean (float): The mean value for the Gaussian distribution.
        stddev (float): The standard deviation for the Gaussian distribution.
        Returns:
        None
        """
        transition_matrix = np.zeros((self.num_cells, self.num_cells))
        
        for i in range(self.num_cells):
            neighbors = self.get_neighbors(i)
            row = np.zeros(self.num_cells)
            for j in neighbors:
                row[j] = np.abs(np.random.normal(mean, stddev))
            row /= row.sum()  # Normalize to make it a probability distribution
            transition_matrix[i] = row
        
        self.add_mobility_pattern(pattern_id, transition_matrix)
    
    def export_cells_to_xml(self, file_path: str):
        """Export cell information to an XML file for SUMO"""
        root = ET.Element("cells")
        for idx, cell in enumerate(self.cells):
            cell_elem = ET.SubElement(root, "cell", id=str(idx), x=str(cell.x), y=str(cell.y))
        
        tree = ET.ElementTree(root)
        tree.write(file_path)
    
    def export_mobility_patterns_to_xml(self, file_path: str):
        """Export mobility patterns to an XML file for SUMO"""
        root = ET.Element("mobilityPatterns")
        for pattern_id, pattern in self.mobility_patterns.items():
            pattern_elem = ET.SubElement(root, "pattern", id=str(pattern_id))
            for i in range(self.num_cells):
                for j in range(self.num_cells):
                    if pattern.transition_matrix[i, j] > 0:
                        ET.SubElement(pattern_elem, "transition", from_cell=str(i), to_cell=str(j), probability=str(pattern.transition_matrix[i, j]))
        
        tree = ET.ElementTree(root)
        tree.write(file_path)