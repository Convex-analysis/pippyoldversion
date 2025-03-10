import os
import re
import json

def remove_paths_from_navigation():
    try:
        # Read paths from all_subsets.txt and clean them
        with open('./MyTest/all_subsets.txt', 'r') as f:
            # Extract just the path part (before the number)
            paths_raw = [line.strip().split(' ')[0] for line in f if line.strip()]
            # Remove trailing slashes for consistent matching
            paths_not_to_remove = set(path.rstrip('/') for path in paths_raw)
        
        print(f"Paths to remove: {paths_not_to_remove}")
        print(f"Number of paths to remove: {len(paths_not_to_remove)}")
        
        # Read navigation instruction list
        with open('./MyTest/navigation_instruction_list.txt', 'r') as f:
            navigation_lines = f.readlines()
        
        # Filter out lines containing paths from all_subsets.txt
        filtered_lines = []
        removed_count = 0
        town_list = [1,6,10]
        
        for line in navigation_lines:
            try:
                # Parse JSON to extract the route_path
                data = json.loads(line)
                route_path = data.get("route_path", "").rstrip('/')
                town_id = data.get("town_id", 0)
                
                if route_path in paths_not_to_remove:
                    filtered_lines.append(line)
                else:
                    removed_count += 1
                    print(f"Removed: {route_path}")
            except json.JSONDecodeError:
                # If line isn't valid JSON, keep it (unlikely in this file)
                filtered_lines.append(line)
        
        # Write back the filtered content
        output_file = os.path.abspath('./MyTest/navigation_instruction_list.txt')
        try:
            with open(output_file, 'w') as f:
                f.writelines(filtered_lines)
                
            print(f"Removed {removed_count} lines from navigation_instruction_list.txt")
            print(f"Remaining lines: {len(filtered_lines)}")
        except PermissionError:
            print(f"Permission denied when writing to {output_file}")
            print("Try closing any programs that might have this file open")
            print("Or run this script with administrator privileges")
            
            # Alternative: Write to a new file instead
            new_file = './MyTest/navigation_instruction_list_new.txt'
            with open(new_file, 'w') as f:
                f.writelines(filtered_lines)
            print(f"Wrote filtered content to {new_file} instead")
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")

if __name__ == "__main__":
    remove_paths_from_navigation()