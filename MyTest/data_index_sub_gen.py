town_list=["town01","town06","town10"]
wheather_list=["w1","w2","w3","w4","w5","w6","w7","w8","w9","w10","w11","w12","w13","w14","w15","w16","w17","w18","w19","w20"]
size = ["long", "short", "tiny"]

def collect_paths_from_file(file_path="dataset_index.txt", n=None):
    """
    Collect paths from the dataset index file.
    
    Args:
        file_path (str): Path to the dataset index file.
        n (int, optional): Maximum number of paths to collect. Default is all paths.
        
    Returns:
        list: A list of collected paths.
    """
    paths = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                path = line.strip()
                if path:  # Skip empty lines
                    paths.append(path)
                    if n is not None and len(paths) >= n:
                        break
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
        
    return paths[:n] if n is not None else paths

def filter_paths_by_criteria(paths, towns=None, weathers=None, sizes=None):
    """
    Filter paths based on town, weather, and size criteria.
    """
    if towns is None and weathers is None and sizes is None:
        return paths
    
    # Fix town_list if it contains a single string with comma-separated values
    if towns and isinstance(towns, list) and len(towns) == 1 and isinstance(towns[0], str) and ',' in towns[0]:
        towns = [t.strip() for t in towns[0].split('_') if t.strip()]
    
    filtered_paths = []
    for path in paths:
        parts = path.split('/')
        #if len(parts) < 2:
            #continue
        
        route_info = parts[0]
        if not route_info.startswith('routes_'):
            continue
        
        # Format: routes_town01_long_w1_08_13_06_50_01
        route_parts = route_info.split('_')
        if len(route_parts) < 4:
            continue
        
        town_part = route_parts[1]  # e.g., town01
        size_part = route_parts[2]  # e.g., long
        weather_part = route_parts[3]  # e.g., w1
        
        # Check if the route matches the criteria
        if towns and town_part not in towns:
            continue
        if sizes and size_part not in sizes:
            continue
        if weathers and weather_part not in weathers:
            continue
        
        filtered_paths.append(path)
    
    return filtered_paths

def generate_subset(file_path="dataset_index.txt", n=None, towns=town_list, weathers=wheather_list, sizes=size):
    """
    Generate a subset of paths from the dataset index file based on criteria.
    
    Args:
        file_path (str): Path to the dataset index file.
        n (int, optional): Maximum number of paths to include.
        towns (list): List of towns to include.
        weathers (list): List of weathers to include.
        sizes (list): List of sizes to include.
        
    Returns:
        list: A subset of paths that meet the criteria.
    """
    all_paths = collect_paths_from_file(file_path)
    filtered_paths = filter_paths_by_criteria(all_paths, towns, weathers, sizes)
    return filtered_paths[:n] if n is not None else filtered_paths