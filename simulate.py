from Cluster.DTMC import DTMC
from Cluster.vehicle import Vehicle
import random

def simulation():
    # Initialize DTMC
    area_range = 20.0
    unit_distance = 1.0
    comm_radius = 5.0
    dtmc = DTMC(area_range, unit_distance, comm_radius)
    # Generate a set of mobility patterns for DTMC
    for pattern_id in range(20):
        dtmc.generate_random_mobility_pattern(pattern_id, mean=0.5, stddev=0.1)

    # Export cells and mobility patterns to XML files for SUMO
    dtmc.export_cells_to_xml("cells.xml")
    dtmc.export_mobility_patterns_to_xml("mobility_patterns.xml")

    # Simulate a vehicle's trajectory in the area
    vehicles = Vehicle.generate_random(num_vehicles=10, gpu_range=(0.5, 1.0), memory_range=(0.5, 1.0), trajectory_length=10, comm_radius=5.0, unit_distance=1.0)
    vehicle1 = vehicles[0]
    initl_cell = random.randint(0, 10)
    current_cell = 0
    next_cell = 0
    for t in range(10):
        if t == 0:
            current_cell = initl_cell
        next_cell = dtmc.predict_next_cell(current_cell, pattern_id)
        neighbors = dtmc.get_neighbors(current_cell)
        print(f"Current cell: {current_cell}")
        
        current_cell = next_cell
        
        print(f"Next cell: {next_cell}")
        print(f"Neighbors: {neighbors}")

def main():
    area_range = 20.0
    unit_distance = 1.0
    comm_radius = 5.0
    pattern_id = 1
    mean = 0.5
    stddev = 0.1
    
    dtmc = DTMC(area_range, unit_distance, comm_radius)
    dtmc.generate_random_mobility_pattern(pattern_id, mean, stddev)
    
    # Example usage
    current_cell = 0
    next_cell = dtmc.predict_next_cell(current_cell, pattern_id)
    neighbors = dtmc.get_neighbors(current_cell)
    
    print(f"Current cell: {current_cell}")
    print(f"Next cell: {next_cell}")
    print(f"Neighbors: {neighbors}")

if __name__ == "__main__":
    simulation()
