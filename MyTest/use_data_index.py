#!/usr/bin/env python3
"""
Script to filter paths from dataset_index.txt according to requirements
and divide them into disjoint subsets with balanced distribution.
"""

import os
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple

from data_index_sub_gen import generate_subset, collect_paths_from_file, filter_paths_by_criteria

def parse_path_info(path: str) -> Tuple[str, str, str]:
    """
    Parse a path to extract town, size, and weather information.
    """
    parts = path.split('/')
    if len(parts) < 2:
        return None, None, None
        
    route_info = parts[0]
    if not route_info.startswith('routes_'):
        return None, None, None
    
    # Format: routes_town01_long_w1_08_13_06_50_01
    route_parts = route_info.split('_')
    if len(route_parts) < 4:
        return None, None, None
    
    town = route_parts[1]  # e.g., town01
    size = route_parts[2]  # e.g., long
    weather = route_parts[3]  # e.g., w1
    
    return town, size, weather

def group_by_attributes(paths: List[str]) -> Dict[Tuple[str, str, str], List[str]]:
    """Group paths by town, size, and weather."""
    groups = defaultdict(list)
    
    for path in paths:
        town, size, weather = parse_path_info(path)
        if town and size and weather:  # Ensure we have valid attributes
            groups[(town, size, weather)].append(path)
    
    return groups

def create_balanced_subsets(paths: List[str], subset_size: int) -> List[List[str]]:
    """
    Create disjoint subsets of specified size, each containing
    all towns, sizes, and weather conditions if possible.
    """
    groups = group_by_attributes(paths)
    
    # Get unique towns, sizes, and weather conditions
    towns = {town for (town, _, _) in groups.keys()}
    sizes = {size for (_, size, _) in groups.keys()}
    weathers = {weather for (_, _, weather) in groups.keys()}
    
    print(f"Found {len(towns)} towns, {len(sizes)} sizes, and {len(weathers)} weather conditions")
    
    # Create balanced subsets
    subsets = []
    remaining_paths = paths.copy()
    random.shuffle(remaining_paths)  # Shuffle to avoid bias
    
    while len(remaining_paths) >= subset_size:
        subset = []
        used_combinations = set()
        
        # First, try to include one item from each town-size-weather combination
        for town in towns:
            for size in sizes:
                for weather in weathers:
                    key = (town, size, weather)
                    # Find paths with this combination that are still available
                    available_paths = [p for p in groups.get(key, []) 
                                      if p in remaining_paths and p not in subset]
                    
                    if available_paths and len(subset) < subset_size:
                        path = random.choice(available_paths)
                        subset.append(path)
                        remaining_paths.remove(path)
                        used_combinations.add(key)
        
        # Fill remaining slots randomly
        while len(subset) < subset_size and remaining_paths:
            path = remaining_paths.pop(0)
            subset.append(path)
        
        if len(subset) == subset_size:
            subsets.append(subset)
        else:
            # If we can't form a complete subset, add remaining paths back
            remaining_paths.extend(subset)
            break
    
    return subsets

def main():
    dataset_index_path = "MyTest/dataset_index.txt"
    
    # Define your requirements here
    requirements = {
        'towns': ['town01', 'town06', 'town10'],
        'sizes': ['tiny', 'short', 'long'],
        'weathers': None  # Use all available weathers
    }
    
    subset_size = 100  # Define your desired subset size N
    
    # Read dataset index and filter paths using functions from data_index_sub_gen.py
    all_paths = collect_paths_from_file(dataset_index_path)
    print(f"Total paths in dataset index: {len(all_paths)}")
    
    # Filter paths according to requirements
    filtered_paths = filter_paths_by_criteria(
        all_paths, 
        towns=requirements['towns'], 
        sizes=requirements['sizes'], 
        weathers=requirements['weathers']
    )
    print(f"Paths after filtering: {len(filtered_paths)}")
    
    # Create disjoint subsets
    subsets = create_balanced_subsets(filtered_paths, subset_size)
    print(f"Created {len(subsets)} subsets of size {subset_size}")
    
    # Save all subsets to a single file
    output_file = "all_subsets.txt"
    with open(output_file, 'w') as f:
        for i, subset in enumerate(subsets):
            #f.write(f"# Subset {i+1} ({len(subset)} paths)\n")
            for path in subset:
                f.write(f"{path}\n")
            #f.write("\n")  # Add an empty line between subsets
    
    print(f"All subsets saved to {output_file}")

if __name__ == "__main__":
    main()
