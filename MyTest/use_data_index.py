#!/usr/bin/env python3
"""
Script to filter paths from dataset_index.txt according to requirements
and divide them into disjoint subsets with balanced distribution.
"""

import os
import random
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple

from data_index_sub_gen import generate_subset, collect_paths_from_file, filter_paths_by_criteria


def parse_path_info(path: str) -> Tuple[str, str, str]:
    """
    Parse a path to extract town, size, and weather information.
    
    Args:
        path (str): A path string like "routes_town01_long_w12_08_13_03_12_20/"
        
    Returns:
        Tuple[str, str, str]: Extracted (town, size, weather)
    """
    # Check if path is valid
    if not path or not isinstance(path, str):
        return None, None, None
        
    # Extract the route part (before any slash)
    route_part = path.split('/')[0] if '/' in path else path
    
    # Check if it's a valid route format
    if not route_part.startswith('routes_'):
        return None, None, None
    
    # Parse parts: routes_town01_long_w12_08_13_03_12_20
    parts = route_part.split('_')
    if len(parts) < 4:
        return None, None, None
    
    try:
        town = parts[1]  # e.g., town01
        size = parts[2]  # e.g., long
        weather = parts[3]  # e.g., w12
        
        # Validate expected formats
        if not town.startswith('town'):
            return None, None, None
        
        if not weather.startswith('w'):
            return None, None, None
            
        return town, size, weather
    except IndexError:
        return None, None, None

def group_by_attributes(paths: List[str]) -> Dict[Tuple[str, str, str], List[str]]:
    """Group paths by town, size, and weather."""
    groups = defaultdict(list)
    
    for path in paths:
        town, size, weather = parse_path_info(path)
        if town and size and weather:  # Ensure we have valid attributes
            groups[(town, size, weather)].append(path)
    
    return groups

def get_path_statistics(paths: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Generate statistics about the paths, including counts by town, size, and weather.
    
    Args:
        paths (List[str]): List of paths to analyze
        
    Returns:
        Dict[str, Dict[str, int]]: Statistics by attribute type
    """
    stats = {
        "town": Counter(),
        "size": Counter(),
        "weather": Counter(),
    }
    
    for path in paths:
        town, size, weather = parse_path_info(path)
        if town and size and weather:
            stats["town"][town] += 1
            stats["size"][size] += 1
            stats["weather"][weather] += 1
    
    return stats

def print_statistics(stats: Dict[str, Dict[str, int]], title: str = "Statistics"):
    """
    Print statistics in a readable format.
    
    Args:
        stats (Dict[str, Dict[str, int]]): Statistics to print
        title (str): Title for the statistics
    """
    print(f"\n{title}:")
    print("=" * 50)
    
    for attr_type, counts in stats.items():
        print(f"\n{attr_type.capitalize()} distribution:")
        print("-" * 30)
        
        # Sort items by count (descending)
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate the total for percentages
        total = sum(counts.values())
        
        # Print each item with its count and percentage
        for item, count in items:
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{item}: {count} paths ({percentage:.2f}%)")
            
    print("=" * 50)

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

def generate_n_subsets(paths: List[str], n_subsets: int, subset_size: int, 
                       towns: List[str], sizes: List[str], weathers: List[str],
                       verbose: bool = True) -> List[List[str]]:
    """
    Generate N subsets from given paths, ensuring each subset contains 
    all combinations of town types and weather types.
    
    Args:
        paths (List[str]): List of all available paths
        n_subsets (int): Number of subsets to generate
        subset_size (int): Size of each subset
        towns (List[str]): List of towns that must be represented in each subset
        sizes (List[str]): List of sizes that should be represented in each subset if possible
        weathers (List[str]): List of weathers that must be represented in each subset
        verbose (bool): Whether to print detailed statistics for each subset
        
    Returns:
        List[List[str]]: N subsets of paths
    """
    # Group paths by their attributes
    groups = group_by_attributes(paths)
    subsets = []
    
    # Track all used paths to maintain disjointness when possible
    used_paths = set()
    
    # Calculate all valid town-weather combinations that we must include
    required_combinations = []
    for town in towns:
        for weather in weathers:
            # For required combinations, consider any size
            town_weather_paths = []
            for size in sizes:
                key = (town, size, weather)
                if key in groups:
                    town_weather_paths.extend(groups[key])
                    
            if town_weather_paths:  # If we have paths for this town-weather combo
                required_combinations.append((town, weather, town_weather_paths))
    
    # Count the total required combinations
    total_required = len(towns) * len(weathers)
    found_required = len(required_combinations)
    
    print(f"Found {found_required} out of {total_required} required town-weather combinations")
    
    if found_required == 0:
        print("Error: No valid town-weather combinations found in the dataset!")
        return []
    
    # Generate each subset
    for subset_idx in range(n_subsets):
        subset = []
        used_paths_this_subset = set()
        
        print(f"\nCreating subset {subset_idx+1}...")
        
        # First, ensure each subset has at least one path from each town-weather combination
        for town, weather, available_paths in required_combinations:
            # First try with unused paths (for disjointness)
            unused_paths = [p for p in available_paths if p not in used_paths and p not in used_paths_this_subset]
            
            if unused_paths:
                # We found an unused path for this town-weather combination
                path = random.choice(unused_paths)
                subset.append(path)
                used_paths_this_subset.add(path)
                used_paths.add(path)
                if verbose:
                    print(f"Added unused path for {town}-{weather}")
            else:
                # No unused paths available, need to reuse a path
                # At least avoid adding duplicates within the same subset
                reusable_paths = [p for p in available_paths if p not in used_paths_this_subset]
                
                if reusable_paths:
                    path = random.choice(reusable_paths)
                    subset.append(path)
                    used_paths_this_subset.add(path)
                    if verbose:
                        print(f"Reused path for {town}-{weather}")
                else:
                    # Last resort: use any path from this combination
                    path = random.choice(available_paths)
                    subset.append(path)
                    used_paths_this_subset.add(path)
                    if verbose:
                        print(f"Had to reuse path for {town}-{weather} that's already in this subset")
        
        # Check how many paths we've added for required combinations
        print(f"Added {len(subset)} paths for required town-weather combinations")
        
        # Try to fill the rest to reach subset_size with unused paths first
        remaining_capacity = subset_size - len(subset)
        if remaining_capacity > 0:
            # First try to include unused paths from all size-town-weather combinations
            available_unused_paths = [p for p in paths if p not in used_paths and p not in used_paths_this_subset]
            random.shuffle(available_unused_paths)
            
            # Add as many unused paths as possible
            for path in available_unused_paths:
                if len(subset) >= subset_size:
                    break
                    
                subset.append(path)
                used_paths_this_subset.add(path)
                used_paths.add(path)
            
            # If we still need more paths, allow reuse from other subsets
            if len(subset) < subset_size:
                print(f"Warning: Subset {subset_idx+1} needs to reuse {subset_size - len(subset)} paths from other subsets")
                remaining_paths = [p for p in paths if p not in used_paths_this_subset]
                random.shuffle(remaining_paths)
                
                while len(subset) < subset_size and remaining_paths:
                    path = remaining_paths.pop(0)
                    subset.append(path)
                    used_paths_this_subset.add(path)
        
        # Add the subset if it's not empty
        if subset:
            subsets.append(subset)
            
            # Verify and print subset coverage
            if verbose or subset_idx == 0:  # Always print first subset to check
                subset_stats = get_path_statistics(subset)
                
                # Check town-weather coverage
                town_weather_covered = 0
                for town in towns:
                    for weather in weathers:
                        has_combo = False
                        for path in subset:
                            t, s, w = parse_path_info(path)
                            if t == town and w == weather:
                                has_combo = True
                                break
                        if has_combo:
                            town_weather_covered += 1
                
                print(f"Subset {subset_idx+1} town-weather coverage: {town_weather_covered}/{len(towns)*len(weathers)} combinations ({town_weather_covered/(len(towns)*len(weathers))*100:.1f}%)")
                
                # Print detailed stats if requested
                if verbose:
                    print_statistics(subset_stats, f"Subset {subset_idx+1} Statistics")
    
    # Print final balance check
    print("\nBalance Check:")
    town_counts = {town: 0 for town in towns}
    weather_counts = {weather: 0 for weather in weathers}
    town_weather_counts = {(town, weather): 0 for town in towns for weather in weathers}
    
    for subset in subsets:
        for path in subset:
            town, size, weather = parse_path_info(path)
            if town in town_counts:
                town_counts[town] += 1
            if weather in weather_counts:
                weather_counts[weather] += 1
            if (town, weather) in town_weather_counts:
                town_weather_counts[(town, weather)] += 1
    
    print("\nTown-Weather combination distribution:")
    for (town, weather), count in sorted(town_weather_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{town}-{weather}: {count} paths")
    
    print("\nTown distribution across all subsets:")
    for town, count in sorted(town_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{town}: {count} paths")
    
    print("\nWeather distribution across all subsets:")
    for weather, count in sorted(weather_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{weather}: {count} paths")
    
    # Check how disjoint the subsets are
    all_paths_in_subsets = []
    for subset in subsets:
        all_paths_in_subsets.extend(subset)
    
    unique_paths = set(all_paths_in_subsets)
    overlap_pct = (1.0 - len(unique_paths)/len(all_paths_in_subsets)) * 100
    
    print(f"\nDisjointness analysis:")
    print(f"Total paths across all subsets: {len(all_paths_in_subsets)}")
    print(f"Unique paths used: {len(unique_paths)}")
    print(f"Path overlap: {overlap_pct:.2f}%")
    
    return subsets

def analyze_subsets(subsets: List[List[str]]) -> Dict[str, list]:
    """
    Analyze the distribution of attributes across all subsets.
    
    Args:
        subsets (List[List[str]]): List of path subsets
        
    Returns:
        Dict[str, list]: Statistics about the subsets
    """
    analysis = {
        "subset_sizes": [],
        "town_coverage": [],
        "size_coverage": [],
        "weather_coverage": [],
        "unique_towns_per_subset": [],
        "unique_sizes_per_subset": [],
        "unique_weathers_per_subset": [],
    }
    
    for i, subset in enumerate(subsets):
        analysis["subset_sizes"].append(len(subset))
        
        # Get statistics for this subset
        stats = get_path_statistics(subset)
        
        # Count unique attributes in this subset
        analysis["unique_towns_per_subset"].append(len(stats["town"]))
        analysis["unique_sizes_per_subset"].append(len(stats["size"]))
        analysis["unique_weathers_per_subset"].append(len(stats["weather"]))
        
        # Store coverage percentage (for future use if needed)
        total_paths = len(subset)
        if total_paths > 0:
            most_common_town = stats["town"].most_common(1)[0] if stats["town"] else (None, 0)
            most_common_size = stats["size"].most_common(1)[0] if stats["size"] else (None, 0)
            most_common_weather = stats["weather"].most_common(1)[0] if stats["weather"] else (None, 0)
            
            analysis["town_coverage"].append((most_common_town[0], most_common_town[1] / total_paths * 100))
            analysis["size_coverage"].append((most_common_size[0], most_common_size[1] / total_paths * 100))
            analysis["weather_coverage"].append((most_common_weather[0], most_common_weather[1] / total_paths * 100))
    
    return analysis

def print_subset_analysis(analysis: Dict[str, list]):
    """
    Print analysis of the generated subsets.
    
    Args:
        analysis (Dict[str, list]): Analysis data to print
    """
    print("\nSubset Analysis Summary:")
    print("=" * 50)
    
    # Calculate average statistics
    avg_size = sum(analysis["subset_sizes"]) / len(analysis["subset_sizes"]) if analysis["subset_sizes"] else 0
    avg_towns = sum(analysis["unique_towns_per_subset"]) / len(analysis["unique_towns_per_subset"]) if analysis["unique_towns_per_subset"] else 0
    avg_sizes = sum(analysis["unique_sizes_per_subset"]) / len(analysis["unique_sizes_per_subset"]) if analysis["unique_sizes_per_subset"] else 0
    avg_weathers = sum(analysis["unique_weathers_per_subset"]) / len(analysis["unique_weathers_per_subset"]) if analysis["unique_weathers_per_subset"] else 0
    
    print(f"Number of subsets: {len(analysis['subset_sizes'])}")
    print(f"Average subset size: {avg_size:.2f} paths")
    print(f"Average number of unique towns per subset: {avg_towns:.2f}")
    print(f"Average number of unique sizes per subset: {avg_sizes:.2f}")
    print(f"Average number of unique weathers per subset: {avg_weathers:.2f}")
    
    # Most common attributes by coverage
    if analysis["town_coverage"]:
        most_common_town = max(analysis["town_coverage"], key=lambda x: x[1])
        print(f"Most represented town across subsets: {most_common_town[0]} (avg {most_common_town[1]:.2f}% coverage)")
        
    if analysis["size_coverage"]:
        most_common_size = max(analysis["size_coverage"], key=lambda x: x[1])
        print(f"Most represented size across subsets: {most_common_size[0]} (avg {most_common_size[1]:.2f}% coverage)")
        
    if analysis["weather_coverage"]:
        most_common_weather = max(analysis["weather_coverage"], key=lambda x: x[1])
        print(f"Most represented weather across subsets: {most_common_weather[0]} (avg {most_common_weather[1]:.2f}% coverage)")
    
    print("=" * 50)

def main():
    # Update dataset path to be more robust
    dataset_index_path = "dataset_index.txt"
    if not os.path.exists(dataset_index_path):
        dataset_index_path = os.path.join("MyTest", "dataset_index.txt")
    if not os.path.exists(dataset_index_path):
        dataset_index_path = os.path.join("d:", "EXP", "pippyoldversion", "MyTest", "dataset_index.txt")
    
    if not os.path.exists(dataset_index_path):
        print(f"Error: Could not find dataset index at {dataset_index_path}")
        print(f"Current working directory is: {os.getcwd()}")
        return
    
    print(f"Using dataset index from: {dataset_index_path}")
    
    # Define your requirements here
    requirements = {
        'towns': ['town01', 'town06', 'town10'],
        'sizes': ['tiny', 'short'],  # Now secondary priority
        'weathers': ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7','w8','w9','w10','w11','w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19','w20']
    }
    
    subset_size = 60  # Keep this value at 60 as requested
    n_subsets = 50     # Number of subsets to generate
    
    # Read dataset index and filter paths using functions from data_index_sub_gen.py
    all_paths = collect_paths_from_file(dataset_index_path)
    print(f"Total paths in dataset index: {len(all_paths)}")
    
    # Generate and print statistics for all paths
    all_paths_stats = get_path_statistics(all_paths)
    print_statistics(all_paths_stats, "All Paths Statistics")
    
    # Filter paths according to requirements
    filtered_paths = filter_paths_by_criteria(
        all_paths, 
        towns=requirements['towns'], 
        sizes=requirements['sizes'], 
        weathers=requirements['weathers']
    )
    print(f"Paths after filtering: {len(filtered_paths)}")
    
    # Generate and print statistics for filtered paths
    filtered_paths_stats = get_path_statistics(filtered_paths)
    print_statistics(filtered_paths_stats, "Filtered Paths Statistics")
    
    # Create N disjoint subsets using the new function
    subsets = generate_n_subsets(
        filtered_paths,
        n_subsets=n_subsets,
        subset_size=subset_size,
        towns=requirements['towns'],
        sizes=requirements['sizes'],
        weathers=requirements['weathers'],
        verbose=False  # Set to True for detailed stats on each subset
    )
    print(f"Created {len(subsets)} subsets of size {subset_size}")
    
    # Analyze the generated subsets
    subset_analysis = analyze_subsets(subsets)
    print_subset_analysis(subset_analysis)
    
    # Save all subsets to a single file
    output_file = "all_subsets.txt"
    with open(output_file, 'w') as f:
        for i, subset in enumerate(subsets):
            # Write only the paths without additional headers/comments
            for path in subset:
                f.write(f"{path}\n")
    
    print(f"All subsets saved to {output_file}")
    
    # Save detailed statistics to a separate file
    stats_file = "subset_statistics.txt"
    with open(stats_file, 'w') as f:
        # Write overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total paths: {len(all_paths)}\n")
        f.write(f"Filtered paths: {len(filtered_paths)}\n")
        f.write(f"Number of subsets: {len(subsets)}\n")
        f.write(f"Subset size: {subset_size}\n\n")
        
        # Calculate and write overall town-weather coverage
        overall_town_weather_coverage = {}
        for town in requirements['towns']:
            for weather in requirements['weathers']:
                overall_town_weather_coverage[(town, weather)] = 0
                
        for subset in subsets:
            for path in subset:
                town, _, weather = parse_path_info(path)
                if (town, weather) in overall_town_weather_coverage:
                    overall_town_weather_coverage[(town, weather)] += 1
        
        f.write("OVERALL TOWN-WEATHER COVERAGE\n")
        f.write("-" * 50 + "\n")
        for (town, weather), count in sorted(overall_town_weather_coverage.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{town}-{weather}: {count} paths\n")
        f.write("\n")
        
        # Write detailed statistics for each subset
        for i, subset in enumerate(subsets):
            stats = get_path_statistics(subset)
            
            f.write(f"SUBSET {i+1} STATISTICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total paths: {len(subset)}\n\n")
            
            # Count town-weather combinations
            town_weather_covered = 0
            town_weather_coverage = {}
            for town in requirements['towns']:
                for weather in requirements['weathers']:
                    has_combo = False
                    for path in subset:
                        t, s, w = parse_path_info(path)
                        if t == town and w == weather:
                            has_combo = True
                            if (town, weather) not in town_weather_coverage:
                                town_weather_coverage[(town, weather)] = 0
                            town_weather_coverage[(town, weather)] += 1
                            break
                    if has_combo:
                        town_weather_covered += 1
            
            f.write(f"Town-Weather Combination Coverage: {town_weather_covered}/{len(requirements['towns'])*len(requirements['weathers'])} combinations\n")
            f.write(f"({town_weather_covered/(len(requirements['towns'])*len(requirements['weathers']))*100:.1f}%)\n\n")
            
            # Write town-weather combination statistics
            f.write("Town-Weather distribution:\n")
            for (town, weather), count in sorted(town_weather_coverage.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {town}-{weather}: {count} paths\n")
            f.write("\n")
            
            # Write town statistics
            f.write("Town distribution:\n")
            for town, count in sorted(stats["town"].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(subset)) * 100
                f.write(f"  {town}: {count} paths ({percentage:.2f}%)\n")
            f.write("\n")
            
            # Write size statistics
            f.write("Size distribution:\n")
            for size, count in sorted(stats["size"].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(subset)) * 100
                f.write(f"  {size}: {count} paths ({percentage:.2f}%)\n")
            f.write("\n")
            
            # Write weather statistics
            f.write("Weather distribution:\n")
            for weather, count in sorted(stats["weather"].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(subset)) * 100
                f.write(f"  {weather}: {count} paths ({percentage:.2f}%)\n")
            f.write("\n\n")
    
    print(f"Detailed statistics saved to {stats_file}")

if __name__ == "__main__":
    main()
