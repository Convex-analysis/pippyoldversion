#!/usr/bin/env python3
"""
Dynamic V2V Channel Emulator

This script emulates dynamic network conditions between vehicles by adjusting
latency and packet loss based on distance traces.
"""

import os
import time
import csv
import argparse
import threading
from typing import Dict, List, Tuple

class NetworkEmulator:
    """Network emulator for simulating V2V channel conditions"""
    
    def __init__(self, csv_file: str, interface: str = "eth0", update_interval: float = 1.0):
        self.csv_file = csv_file
        self.interface = interface
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.trace_data = []
        
    def load_trace(self) -> None:
        """Load distance trace from CSV file"""
        self.trace_data = []
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = float(row['timestamp'])
                distance = float(row['distance'])
                self.trace_data.append((timestamp, distance))
        print(f"Loaded {len(self.trace_data)} trace points")
    
    def calculate_network_params(self, distance: float) -> Tuple[float, float]:
        """Calculate latency and packet loss based on distance"""
        # Simple model: latency increases with distance, packet loss increases exponentially
        base_latency = 1.0  # ms
        distance_factor = distance / 100.0  # Normalize distance
        
        latency = base_latency + (distance_factor * 5.0)  # Up to 5ms per 100m
        packet_loss = min(100.0, (distance_factor ** 2) * 10.0)  # Up to 10% packet loss at 100m
        
        return latency, packet_loss
    
    def set_network_params(self, latency: float, packet_loss: float) -> None:
        """Set network parameters using tc qdisc"""
        try:
            # Clear existing qdisc
            os.system(f"tc qdisc del dev {self.interface} root 2>/dev/null")
            
            # Set new qdisc with netem
            command = f"tc qdisc add dev {self.interface} root netem delay {latency:.1f}ms loss {packet_loss:.1f}%"
            os.system(command)
            print(f"Updated network parameters: latency={latency:.1f}ms, loss={packet_loss:.1f}%")
        except Exception as e:
            print(f"Error setting network parameters: {e}")
    
    def run(self) -> None:
        """Main emulation loop"""
        start_time = time.time()
        trace_index = 0
        
        while self.running and trace_index < len(self.trace_data):
            current_time = time.time() - start_time
            
            # Find the current trace point
            while trace_index < len(self.trace_data) and self.trace_data[trace_index][0] <= current_time:
                trace_index += 1
            
            if trace_index > 0:
                _, distance = self.trace_data[trace_index - 1]
                latency, packet_loss = self.calculate_network_params(distance)
                self.set_network_params(latency, packet_loss)
            
            time.sleep(self.update_interval)
        
        # Reset network parameters when done
        self.reset_network()
    
    def reset_network(self) -> None:
        """Reset network parameters to default"""
        try:
            os.system(f"tc qdisc del dev {self.interface} root 2>/dev/null")
            os.system(f"tc qdisc add dev {self.interface} root pfifo_fast")
            print("Network parameters reset to default")
        except Exception as e:
            print(f"Error resetting network parameters: {e}")
    
    def start(self) -> None:
        """Start the network emulator"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            print("Network emulator started")
    
    def stop(self) -> None:
        """Stop the network emulator"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            print("Network emulator stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dynamic V2V Channel Emulator")
    parser.add_argument("csv_file", help="CSV file with distance trace")
    parser.add_argument("--interface", default="eth0", help="Network interface to modify")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    emulator = NetworkEmulator(args.csv_file, args.interface, args.interval)
    emulator.load_trace()
    emulator.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping network emulator...")
        emulator.stop()

if __name__ == "__main__":
    main()