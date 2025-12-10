"""
Neighbor Discovery and V2V Communication Protocol for FHDP System

Handles vehicle-to-vehicle communication, neighbor discovery, and message
routing with support for multiple V2V protocols (DSRC, C-V2X, WiFi-Direct).
"""
import time
import threading
import socket
import struct
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import queue

from ..core.types import (
    VehicleInfo, CommunicationProtocol, CommunicationBundle,
    ModelUpdate, Pipeline, ErrorPropagation
)
from ..core.constants import (
    MAX_NEIGHBOR_DISTANCE, MIN_SIGNAL_STRENGTH,
    PROTOCOL_BANDWIDTH, PROTOCOL_LATENCY, PROTOCOL_RANGE,
    COMMUNICATION_BUNDLE_SIZE, MAX_CONNECTION_ATTEMPTS,
    CONNECTION_TIMEOUT, HEARTBEAT_INTERVAL, MISSING_HEARTBEAT_THRESHOLD
)

@dataclass
class NeighborInfo:
    """Information about a neighboring vehicle"""
    vehicle_id: str
    position: Tuple[float, float]
    velocity: float
    direction: float
    protocol: CommunicationProtocol
    signal_strength: float  # dBm
    bandwidth: float  # Mbps
    latency: float  # seconds
    last_seen: float
    connection_quality: float  # 0.0-1.0
    
@dataclass
class V2VMessage:
    """V2V communication message"""
    message_id: str
    sender_id: str
    receiver_id: str  # 'broadcast' for broadcast messages
    message_type: str  # 'discovery', 'heartbeat', 'data', 'pipeline', 'model_update'
    payload: Dict[str, Any]
    timestamp: float
    protocol: CommunicationProtocol
    ttl: int = 3  # Time to live for message forwarding
    
class ProtocolManager:
    """Manages different V2V communication protocols"""
    
    def __init__(self):
        self.active_protocols: Dict[CommunicationProtocol, bool] = {
            CommunicationProtocol.DSRC: True,
            CommunicationProtocol.CV2X: False,
            CommunicationProtocol.WIFI_DIRECT: False
        }
        self.protocol_sockets: Dict[CommunicationProtocol, socket.socket] = {}
        self.protocol_listeners: Dict[CommunicationProtocol, threading.Thread] = {}
        
    def enable_protocol(self, protocol: CommunicationProtocol, port: int = 0) -> bool:
        """Enable a communication protocol"""
        if protocol in self.protocol_sockets:
            return True  # Already enabled
        
        try:
            # Create socket for protocol
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(1.0)
            
            # Bind to port
            if port == 0:
                port = self._get_default_port(protocol)
            
            sock.bind(('', port))
            
            self.protocol_sockets[protocol] = sock
            self.active_protocols[protocol] = True
            
            # Start listener thread
            listener = threading.Thread(target=self._listen_for_messages, args=(protocol,), daemon=True)
            listener.start()
            self.protocol_listeners[protocol] = listener
            
            return True
            
        except Exception as e:
            print(f"Failed to enable protocol {protocol}: {e}")
            return False
    
    def _get_default_port(self, protocol: CommunicationProtocol) -> int:
        """Get default port for protocol"""
        port_map = {
            CommunicationProtocol.DSRC: 20001,
            CommunicationProtocol.CV2X: 20002,
            CommunicationProtocol.WIFI_DIRECT: 20003
        }
        return port_map.get(protocol, 20001)
    
    def _listen_for_messages(self, protocol: CommunicationProtocol):
        """Listen for incoming messages on protocol"""
        sock = self.protocol_sockets[protocol]
        
        while self.active_protocols.get(protocol, False):
            try:
                data, addr = sock.recvfrom(65535)
                message = self._deserialize_message(data)
                
                if message and message.receiver_id in ['broadcast', self._get_vehicle_id()]:
                    self._handle_incoming_message(message, addr)
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error listening on {protocol}: {e}")
    
    def _get_vehicle_id(self) -> str:
        """Get current vehicle ID (placeholder)"""
        # This would be provided by the vehicle system
        return "current_vehicle"
    
    def _deserialize_message(self, data: bytes) -> Optional[V2VMessage]:
        """Deserialize message from bytes"""
        try:
            # Simple JSON serialization for now
            message_dict = json.loads(data.decode('utf-8'))
            return V2VMessage(
                message_id=message_dict['message_id'],
                sender_id=message_dict['sender_id'],
                receiver_id=message_dict['receiver_id'],
                message_type=message_dict['message_type'],
                payload=message_dict['payload'],
                timestamp=message_dict['timestamp'],
                protocol=CommunicationProtocol(message_dict['protocol'])
            )
        except Exception as e:
            print(f"Failed to deserialize message: {e}")
            return None
    
    def _handle_incoming_message(self, message: V2VMessage, addr: Tuple[str, int]):
        """Handle incoming message (to be overridden by subclasses)"""
        pass
    
    def send_message(self, message: V2VMessage, target_address: Optional[Tuple[str, int]] = None) -> bool:
        """Send message using specified protocol"""
        if message.protocol not in self.protocol_sockets:
            return False
        
        try:
            sock = self.protocol_sockets[message.protocol]
            
            # Serialize message
            message_data = self._serialize_message(message)
            
            if target_address:
                # Send to specific address
                sock.sendto(message_data, target_address)
            else:
                # Broadcast (simplified - would need proper broadcast mechanism)
                sock.sendto(message_data, ('<broadcast>', self._get_default_port(message.protocol)))
            
            return True
            
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    def _serialize_message(self, message: V2VMessage) -> bytes:
        """Serialize message to bytes"""
        message_dict = {
            'message_id': message.message_id,
            'sender_id': message.sender_id,
            'receiver_id': message.receiver_id,
            'message_type': message.message_type,
            'payload': message.payload,
            'timestamp': message.timestamp,
            'protocol': message.protocol.value
        }
        return json.dumps(message_dict).encode('utf-8')
    
    def disable_protocol(self, protocol: CommunicationProtocol):
        """Disable a communication protocol"""
        self.active_protocols[protocol] = False
        
        if protocol in self.protocol_sockets:
            self.protocol_sockets[protocol].close()
            del self.protocol_sockets[protocol]
        
        if protocol in self.protocol_listeners:
            self.protocol_listeners[protocol].join(timeout=2.0)
            del self.protocol_listeners[protocol]

class NeighborDiscovery:
    """Handles neighbor vehicle discovery using periodic beacons"""
    
    def __init__(self, protocol_manager: ProtocolManager, vehicle_info: VehicleInfo):
        self.protocol_manager = protocol_manager
        self.vehicle_info = vehicle_info
        self.neighbors: Dict[str, NeighborInfo] = {}
        self.discovery_callbacks: List[Callable[[str, NeighborInfo], None]] = []
        
        # Discovery parameters
        self.discovery_interval = 2.0  # seconds
        self.neighbor_timeout = 10.0  # seconds
        
        # Threading
        self.discovery_thread = None
        self.cleanup_thread = None
        self.stop_event = threading.Event()
        
    def start_discovery(self):
        """Start neighbor discovery process"""
        if self.discovery_thread:
            return
        
        self.stop_event.clear()
        
        # Start discovery thread
        self.discovery_thread = threading.Thread(target=self._discovery_worker, daemon=True)
        self.discovery_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def stop_discovery(self):
        """Stop neighbor discovery process"""
        self.stop_event.set()
        
        if self.discovery_thread:
            self.discovery_thread.join(timeout=2.0)
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=2.0)
    
    def _discovery_worker(self):
        """Discovery worker thread"""
        while not self.stop_event.is_set():
            try:
                self._send_discovery_beacon()
                time.sleep(self.discovery_interval)
            except Exception as e:
                print(f"Discovery error: {e}")
    
    def _send_discovery_beacon(self):
        """Send discovery beacon message"""
        for protocol in CommunicationProtocol:
            if self.protocol_manager.active_protocols.get(protocol, False):
                beacon = V2VMessage(
                    message_id=f"beacon_{int(time.time() * 1000)}",
                    sender_id=self.vehicle_info.vehicle_id,
                    receiver_id="broadcast",
                    message_type="discovery",
                    payload={
                        'position': self.vehicle_info.position,
                        'velocity': self.vehicle_info.velocity,
                        'direction': self.vehicle_info.direction,
                        'resources': self.vehicle_info.resources
                    },
                    timestamp=time.time(),
                    protocol=protocol
                )
                
                self.protocol_manager.send_message(beacon)
    
    def handle_discovery_message(self, message: V2VMessage, signal_strength: float = -70.0):
        """Handle incoming discovery message"""
        if message.sender_id == self.vehicle_info.vehicle_id:
            return  # Ignore self
        
        # Calculate distance
        payload = message.payload
        neighbor_pos = payload.get('position', (0, 0))
        distance = np.sqrt(
            (self.vehicle_info.position[0] - neighbor_pos[0]) ** 2 +
            (self.vehicle_info.position[1] - neighbor_pos[1]) ** 2
        )
        
        if distance > MAX_NEIGHBOR_DISTANCE:
            return  # Too far away
        
        # Create neighbor info
        neighbor = NeighborInfo(
            vehicle_id=message.sender_id,
            position=neighbor_pos,
            velocity=payload.get('velocity', 0.0),
            direction=payload.get('direction', 0.0),
            protocol=message.protocol,
            signal_strength=signal_strength,
            bandwidth=PROTOCOL_BANDWIDTH[message.protocol.value],
            latency=PROTOCOL_LATENCY[message.protocol.value],
            last_seen=message.timestamp,
            connection_quality=self._calculate_connection_quality(signal_strength, distance)
        )
        
        # Update neighbors
        old_neighbor = self.neighbors.get(message.sender_id)
        self.neighbors[message.sender_id] = neighbor
        
        # Notify callbacks if new neighbor or significant change
        if (old_neighbor is None or 
            abs(old_neighbor.connection_quality - neighbor.connection_quality) > 0.2):
            for callback in self.discovery_callbacks:
                try:
                    callback(message.sender_id, neighbor)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    def _calculate_connection_quality(self, signal_strength: float, distance: float) -> float:
        """Calculate connection quality based on signal strength and distance"""
        # Signal strength factor (higher is better)
        signal_factor = max(0.0, (signal_strength - MIN_SIGNAL_STRENGTH) / 
                          (abs(MIN_SIGNAL_STRENGTH) + 20.0))
        
        # Distance factor (closer is better)
        distance_factor = max(0.0, 1.0 - distance / MAX_NEIGHBOR_DISTANCE)
        
        return (signal_factor * 0.6 + distance_factor * 0.4)
    
    def _cleanup_worker(self):
        """Cleanup worker thread"""
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                expired_neighbors = []
                
                for vehicle_id, neighbor in self.neighbors.items():
                    if current_time - neighbor.last_seen > self.neighbor_timeout:
                        expired_neighbors.append(vehicle_id)
                
                for vehicle_id in expired_neighbors:
                    del self.neighbors[vehicle_id]
                    # Notify callbacks about neighbor departure
                    for callback in self.discovery_callbacks:
                        try:
                            callback(vehicle_id, None)
                        except Exception as e:
                            print(f"Callback error: {e}")
                
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    def add_discovery_callback(self, callback: Callable[[str, Optional[NeighborInfo]], None]):
        """Add callback for neighbor discovery events"""
        self.discovery_callbacks.append(callback)
    
    def get_neighbors(self) -> Dict[str, NeighborInfo]:
        """Get current neighbors"""
        return self.neighbors.copy()
    
    def get_neighbors_by_protocol(self, protocol: CommunicationProtocol) -> Dict[str, NeighborInfo]:
        """Get neighbors using specific protocol"""
        return {k: v for k, v in self.neighbors.items() if v.protocol == protocol}
    
    def get_best_neighbors(self, count: int = 5) -> List[Tuple[str, NeighborInfo]]:
        """Get best neighbors by connection quality"""
        sorted_neighbors = sorted(
            self.neighbors.items(),
            key=lambda x: x[1].connection_quality,
            reverse=True
        )
        return sorted_neighbors[:count]

class MessageRouter:
    """Routes messages between vehicles with load balancing and reliability"""
    
    def __init__(self, protocol_manager: ProtocolManager):
        self.protocol_manager = protocol_manager
        self.message_queue = queue.Queue()
        self.routing_table: Dict[str, Tuple[str, CommunicationProtocol]] = {}  # vehicle_id -> (address, protocol)
        self.message_handlers: Dict[str, Callable] = {}
        
        # Threading
        self.router_thread = None
        self.stop_event = threading.Event()
        
        # Message statistics
        self.message_stats = {
            'sent': 0,
            'received': 0,
            'failed': 0,
            'bundled': 0
        }
    
    def start_routing(self):
        """Start message routing service"""
        if self.router_thread:
            return
        
        self.stop_event.clear()
        self.router_thread = threading.Thread(target=self._routing_worker, daemon=True)
        self.router_thread.start()
    
    def stop_routing(self):
        """Stop message routing service"""
        self.stop_event.set()
        
        if self.router_thread:
            self.router_thread.join(timeout=2.0)
    
    def _routing_worker(self):
        """Message routing worker"""
        while not self.stop_event.is_set():
            try:
                message, target_address = self.message_queue.get(timeout=1.0)
                success = self._route_message(message, target_address)
                
                if success:
                    self.message_stats['sent'] += 1
                else:
                    self.message_stats['failed'] += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Routing error: {e}")
    
    def send_message(self, message: V2VMessage, target_vehicle_id: str = None):
        """Send message to target vehicle"""
        if target_vehicle_id and target_vehicle_id in self.routing_table:
            address, protocol = self.routing_table[target_vehicle_id]
            target_address = (address, self.protocol_manager._get_default_port(protocol))
        else:
            target_address = None  # Broadcast
        
        self.message_queue.put((message, target_address))
    
    def _route_message(self, message: V2VMessage, target_address: Optional[Tuple[str, int]]) -> bool:
        """Route message to destination"""
        return self.protocol_manager.send_message(message, target_address)
    
    def send_bundle(self, bundle: CommunicationBundle, target_vehicle_id: str = None):
        """Send message bundle for efficiency"""
        if target_vehicle_id and target_vehicle_id in self.routing_table:
            address, protocol = self.routing_table[target_vehicle_id]
        else:
            # Choose best protocol
            protocol = CommunicationProtocol.DSRC  # Default
            address = "<broadcast>"
        
        # Create bundle message
        bundle_message = V2VMessage(
            message_id=f"bundle_{int(time.time() * 1000)}",
            sender_id=self.protocol_manager._get_vehicle_id(),
            receiver_id=target_vehicle_id or "broadcast",
            message_type="bundle",
            payload={
                'messages': bundle.messages,
                'compression_ratio': bundle.compression_ratio
            },
            timestamp=time.time(),
            protocol=protocol
        )
        
        self.message_stats['bundled'] += 1
        return self._route_message(bundle_message, 
                                  (address, self.protocol_manager._get_default_port(protocol)) if address != "<broadcast>" else None)
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler
    
    def handle_incoming_message(self, message: V2VMessage, sender_address: Tuple[str, int]):
        """Handle incoming message"""
        # Update routing table
        self.routing_table[message.sender_id] = (sender_address[0], message.protocol)
        
        # Call appropriate handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                handler(message, sender_address)
            except Exception as e:
                print(f"Handler error: {e}")
        
        self.message_stats['received'] += 1
    
    def get_routing_statistics(self) -> Dict[str, int]:
        """Get routing statistics"""
        return self.message_stats.copy()

class V2VCommunicationManager:
    """Main V2V communication manager"""
    
    def __init__(self, vehicle_info: VehicleInfo):
        self.vehicle_info = vehicle_info
        self.protocol_manager = ProtocolManager()
        self.neighbor_discovery = NeighborDiscovery(self.protocol_manager, vehicle_info)
        self.message_router = MessageRouter(self.protocol_manager)
        
        # Communication state
        self.is_active = False
        self.active_neighbors: Set[str] = set()
        
        # Callbacks
        self.neighbor_callbacks: List[Callable] = []
        self.message_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
    def initialize(self, protocols: List[CommunicationProtocol] = None):
        """Initialize V2V communication"""
        if protocols is None:
            protocols = [CommunicationProtocol.DSRC]
        
        # Enable protocols
        for protocol in protocols:
            self.protocol_manager.enable_protocol(protocol)
        
        # Set up message handling
        self._setup_message_handling()
        
        # Start services
        self.message_router.start_routing()
        self.neighbor_discovery.start_discovery()
        
        self.is_active = True
    
    def _setup_message_handling(self):
        """Set up message handling"""
        # Register message handlers
        self.message_router.register_handler("discovery", self._handle_discovery_message)
        self.message_router.register_handler("heartbeat", self._handle_heartbeat_message)
        self.message_router.register_handler("data", self._handle_data_message)
        self.message_router.register_handler("pipeline", self._handle_pipeline_message)
        self.message_router.register_handler("model_update", self._handle_model_update)
        
        # Set up neighbor discovery callback
        self.neighbor_discovery.add_discovery_callback(self._on_neighbor_update)
    
    def _handle_discovery_message(self, message: V2VMessage, sender_address: Tuple[str, int]):
        """Handle discovery message"""
        self.neighbor_discovery.handle_discovery_message(message)
    
    def _handle_heartbeat_message(self, message: V2VMessage, sender_address: Tuple[str, int]):
        """Handle heartbeat message"""
        # Update neighbor last seen time
        neighbors = self.neighbor_discovery.get_neighbors()
        if message.sender_id in neighbors:
            neighbors[message.sender_id].last_seen = message.timestamp
    
    def _handle_data_message(self, message: V2VMessage, sender_address: Tuple[str, int]):
        """Handle data message"""
        for callback in self.message_callbacks["data"]:
            callback(message, sender_address)
    
    def _handle_pipeline_message(self, message: V2VMessage, sender_address: Tuple[str, int]):
        """Handle pipeline-related message"""
        for callback in self.message_callbacks["pipeline"]:
            callback(message, sender_address)
    
    def _handle_model_update(self, message: V2VMessage, sender_address: Tuple[str, int]):
        """Handle model update message"""
        for callback in self.message_callbacks["model_update"]:
            callback(message, sender_address)
    
    def _on_neighbor_update(self, vehicle_id: str, neighbor_info: Optional[NeighborInfo]):
        """Handle neighbor update"""
        if neighbor_info:
            self.active_neighbors.add(vehicle_id)
        else:
            self.active_neighbors.discard(vehicle_id)
        
        for callback in self.neighbor_callbacks:
            callback(vehicle_id, neighbor_info)
    
    def send_model_update(self, target_id: str, model_update: ModelUpdate):
        """Send model update to target vehicle"""
        message = V2VMessage(
            message_id=f"model_update_{int(time.time() * 1000)}",
            sender_id=self.vehicle_info.vehicle_id,
            receiver_id=target_id,
            message_type="model_update",
            payload={
                'model_data': model_update.update_data,
                'metadata': model_update.metadata,
                'training_mode': model_update.training_mode.value,
                'fidelity_score': model_update.fidelity_score
            },
            timestamp=time.time(),
            protocol=CommunicationProtocol.DSRC  # Choose best protocol
        )
        
        self.message_router.send_message(message, target_id)
    
    def send_pipeline_invitation(self, target_vehicles: List[str], pipeline_info: Dict[str, Any]):
        """Send pipeline formation invitation"""
        for target_id in target_vehicles:
            message = V2VMessage(
                message_id=f"pipeline_invite_{int(time.time() * 1000)}",
                sender_id=self.vehicle_info.vehicle_id,
                receiver_id=target_id,
                message_type="pipeline",
                payload={
                    'type': 'invitation',
                    'pipeline_info': pipeline_info
                },
                timestamp=time.time(),
                protocol=CommunicationProtocol.DSRC
            )
            
            self.message_router.send_message(message, target_id)
    
    def broadcast_heartbeat(self):
        """Broadcast heartbeat message"""
        message = V2VMessage(
            message_id=f"heartbeat_{int(time.time() * 1000)}",
            sender_id=self.vehicle_info.vehicle_id,
            receiver_id="broadcast",
            message_type="heartbeat",
            payload={
                'status': 'active',
                'resources': self.vehicle_info.resources
            },
            timestamp=time.time(),
            protocol=CommunicationProtocol.DSRC
        )
        
        self.message_router.send_message(message)
    
    def register_neighbor_callback(self, callback: Callable):
        """Register neighbor update callback"""
        self.neighbor_callbacks.append(callback)
    
    def register_message_callback(self, message_type: str, callback: Callable):
        """Register message callback"""
        self.message_callbacks[message_type].append(callback)
    
    def get_neighbors(self) -> Dict[str, NeighborInfo]:
        """Get current neighbors"""
        return self.neighbor_discovery.get_neighbors()
    
    def get_active_neighbors(self) -> Set[str]:
        """Get active neighbor IDs"""
        return self.active_neighbors.copy()
    
    def get_best_neighbors(self, count: int = 5) -> List[Tuple[str, NeighborInfo]]:
        """Get best neighbors for communication"""
        return self.neighbor_discovery.get_best_neighbors(count)
    
    def shutdown(self):
        """Shutdown V2V communication"""
        if not self.is_active:
            return
        
        self.neighbor_discovery.stop_discovery()
        self.message_router.stop_routing()
        
        # Disable all protocols
        for protocol in CommunicationProtocol:
            self.protocol_manager.disable_protocol(protocol)
        
        self.is_active = False