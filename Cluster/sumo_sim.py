from Cluster.DTMC import DTMC
import os
import subprocess

def generate_sumo_scenario():
    # 1. Initialize DTMC with 20x20 grid
    area_range = 20.0  # This will create a 20x20 grid since unit_distance is 1.0
    unit_distance = 1.0
    comm_radius = 5.0
    dtmc = DTMC(area_range, unit_distance, comm_radius)

    # 2. Generate mobility patterns
    num_patterns = 5
    for pattern_id in range(num_patterns):
        dtmc.generate_random_mobility_pattern(
            pattern_id=pattern_id,
            mean=0.5,
            stddev=0.1
        )

    # 3. Export DTMC data to XML files
    dtmc.export_cells_to_xml("sumo_files/cells.xml")
    dtmc.export_mobility_patterns_to_xml("sumo_files/mobility_patterns.xml")

    # 4. Create SUMO network file
    with open("sumo_files/network.nod.xml", "w") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <!-- Create nodes for each cell -->
""")
        grid_size = int(area_range)
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = f"node_{i}_{j}"
                x = i * 100  # Scale up for SUMO visualization
                y = j * 100
                f.write(f'    <node id="{node_id}" x="{x}" y="{y}" type="priority"/>\n')
        f.write("</nodes>")

    # 5. Create SUMO edges file
    with open("sumo_files/network.edg.xml", "w") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <!-- Create edges between adjacent nodes -->
""")
        grid_size = int(area_range)
        for i in range(grid_size):
            for j in range(grid_size):
                # Horizontal edges
                if i < grid_size - 1:
                    from_node = f"node_{i}_{j}"
                    to_node = f"node_{i+1}_{j}"
                    edge_id = f"edge_{i}_{j}_h"
                    f.write(f'    <edge id="{edge_id}" from="{from_node}" to="{to_node}" numLanes="2" speed="13.89"/>\n')
                
                # Vertical edges
                if j < grid_size - 1:
                    from_node = f"node_{i}_{j}"
                    to_node = f"node_{i}_{j+1}"
                    edge_id = f"edge_{i}_{j}_v"
                    f.write(f'    <edge id="{edge_id}" from="{from_node}" to="{to_node}" numLanes="2" speed="13.89"/>\n')
        f.write("</edges>")

    # 6. Generate SUMO network using netconvert
    subprocess.run([
        "netconvert",
        "--node-files=sumo_files/network.nod.xml",
        "--edge-files=sumo_files/network.edg.xml",
        "--output-file=sumo_files/network.net.xml"
    ])

    # 7. Create SUMO route file based on mobility patterns
    with open("sumo_files/routes.rou.xml", "w") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15"/>
""")
        # Generate routes based on mobility patterns
        for pattern_id in range(num_patterns):
            route_id = f"route_{pattern_id}"
            f.write(f'    <route id="{route_id}" edges="edge_0_0_h edge_0_0_v"/>\n')
        
        # Generate vehicles
        for i in range(10):  # Generate 10 vehicles
            f.write(f'    <vehicle id="vehicle_{i}" type="car" route="route_0" depart="{i}"/>\n')
        f.write("</routes>")

    # 8. Create SUMO configuration file
    with open("sumo_files/scenario.sumocfg", "w") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>
</configuration>
""")

if __name__ == "__main__":
    # Create directory for SUMO files if it doesn't exist
    os.makedirs("sumo_files", exist_ok=True)
    generate_sumo_scenario()
    print("SUMO scenario generated successfully!")