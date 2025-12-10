"""
Cross-Platform Communication Protocol for FHDP

Optimized communication protocols that work across Jetson Orin Nano, x86 PCs,
and other heterogeneous computing platforms.
"""
import asyncio
import socket
import json
import time
import threading
import ssl
import zlib
import pickle
import hashlib
import glob
import psutil
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import queue
from concurrent.futures import ThreadPoolExecutor

from .hardware_adapter import HardwarePlatform, HardwareCapabilities, NetworkInterface
from .types import CommunicationBundle, CommunicationProtocol

class TransportProtocol(Enum):
    """Transport layer protocols"""
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MPI = "mpi"
    RDMA = "rdma"

class CompressionType(Enum):
    """Compression algorithms"""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    BROTLI = "brotli"

@dataclass
class NetworkEndpoint:
    """Network endpoint configuration"""
    host: str
    port: int
    protocol: TransportProtocol
    ssl_enabled: bool = False
    ssl_context: Optional[ssl.SSLContext] = None
    compression: CompressionType = CompressionType.ZLIB

@dataclass
class MessageMetrics:
    """Message transmission metrics"""
    message_id: str
    source_id: str
    target_id: str
    timestamp: float
    size_bytes: int
    compression_ratio: float
    transmission_time: float
    success: bool
    protocol: TransportProtocol
    retry_count: int = 0

@dataclass 
class CrossPlatformMessage:
    """Cross-platform message format"""
    message_id: str
    source_id: str
    target_id: str
    message_type: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: float = 30.0  # Time to live in seconds
    requires_ack: bool = True
    priority: int = 0  # Higher = more important
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrossPlatformMessage':
        """Create message from dictionary"""
        return cls(**data)

class CompressionManager:
    """Manages compression algorithms"""
    
    def __init__(self):
        self.compression_map = {
            CompressionType.NONE: self._no_compress,
            CompressionType.ZLIB: self._zlib_compress,
            # CompressionType.LZ4: self._lz4_compress,
            # CompressionType.BROTLI: self._brotli_compress,
        }
        self.decompression_map = {
            CompressionType.NONE: self._no_decompress,
            CompressionType.ZLIB: self._zlib_decompress,
            # CompressionType.LZ4: self._lz4_decompress,
            # CompressionType.BROTLI: self._brotli_decompress,
        }
    
    def compress(self, data: bytes, compression_type: CompressionType) -> Tuple[bytes, float]:
        """Compress data"""
        compress_func = self.compression_map.get(compression_type, self._no_compress)
        start_time = time.time()
        compressed = compress_func(data)
        compression_time = time.time() - start_time
        
        if len(data) > 0:
            compression_ratio = len(compressed) / len(data)
        else:
            compression_ratio = 1.0
            
        return compressed, compression_ratio
    
    def decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data"""
        decompress_func = self.decompression_map.get(compression_type, self._no_decompress)
        return decompress_func(data)
    
    def _no_compress(self, data: bytes) -> bytes:
        return data
    
    def _no_decompress(self, data: bytes) -> bytes:
        return data
    
    def _zlib_compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=6)
    
    def _zlib_decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)

class ConnectionPool:
    """Manages network connections efficiently"""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.active_connections: Dict[str, socket.socket] = {}
        self.connection_lock = threading.Lock()
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = 30.0
        self.last_cleanup = time.time()
        
    def get_connection(self, endpoint: NetworkEndpoint) -> Optional[socket.socket]:
        """Get or create connection"""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"
        
        with self.connection_lock:
            # Check existing connection
            if connection_key in self.active_connections:
                conn = self.active_connections[connection_key]
                if self._is_connection_healthy(conn):
                    return conn
                else:
                    # Remove dead connection
                    try:
                        conn.close()
                    except:
                        pass
                    del self.active_connections[connection_key]
            
            # Create new connection
            if len(self.active_connections) >= self.max_connections:
                self._cleanup_connections()
            
            new_conn = self._create_connection(endpoint)
            if new_conn:
                self.active_connections[connection_key] = new_conn
                self.connection_stats[connection_key] = {
                    'created': time.time(),
                    'last_used': time.time(),
                    'bytes_sent': 0,
                    'bytes_received': 0,
                    'messages_sent': 0,
                    'messages_received': 0
                }
            
            return new_conn
    
    def _is_connection_healthy(self, conn: socket.socket) -> bool:
        """Check if connection is healthy"""
        try:
            # Try a simple peek
            conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            return False  # If there's data, connection might be stale
        except BlockingIOError:
            # No data waiting, connection is likely healthy
            return True
        except:
            return False
    
    def _create_connection(self, endpoint: NetworkEndpoint) -> Optional[socket.socket]:
        """Create new connection"""
        try:
            if endpoint.protocol == TransportProtocol.TCP:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(10.0)  # 10 second timeout
                
                if endpoint.ssl_enabled and endpoint.ssl_context:
                    conn = endpoint.ssl_context.wrap_socket(conn, server_hostname=endpoint.host)
                
                conn.connect((endpoint.host, endpoint.port))
                return conn
            
            elif endpoint.protocol == TransportProtocol.UDP:
                conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                return conn
            
            else:
                logging.warning(f"Unsupported protocol: {endpoint.protocol}")
                return None
                
        except Exception as e:
            logging.error(f"Failed to create connection: {e}")
            return None
    
    def release_connection(self, endpoint: NetworkEndpoint):
        """Release connection back to pool"""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"
        
        with self.connection_lock:
            if connection_key in self.connection_stats:
                self.connection_stats[connection_key]['last_used'] = time.time()
    
    def _cleanup_connections(self):
        """Clean up old connections"""
        current_time = time.time()
        connections_to_remove = []
        
        for key, conn in self.active_connections.items():
            stats = self.connection_stats.get(key, {})
            last_used = stats.get('last_used', 0)
            
            # Remove connections unused for 5 minutes
            if current_time - last_used > 300:
                connections_to_remove.append(key)
                try:
                    conn.close()
                except:
                    pass
        
        for key in connections_to_remove:
            del self.active_connections[key]
            if key in self.connection_stats:
                del self.connection_stats[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'active_connections': len(self.active_connections),
            'max_connections': self.max_connections,
            'connection_stats': dict(self.connection_stats)
        }

class MessageRouter:
    """Routes messages between different platforms"""
    
    def __init__(self, hardware_capabilities: HardwareCapabilities):
        self.hardware_capabilities = hardware_capabilities
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.routing_table: Dict[str, NetworkEndpoint] = {}
        self.message_queue = queue.Queue()
        self.metrics: List[MessageMetrics] = []
        self.compression_manager = CompressionManager()
        self.connection_pool = ConnectionPool()
        
        # Platform-specific optimizations
        self.optimize_for_platform()
    
    def optimize_for_platform(self):
        """Apply platform-specific optimizations"""
        platform = self.hardware_capabilities.platform
        
        if platform == HardwarePlatform.JETSON_ORIN:
            # Jetson Orin optimizations: Use compression to save bandwidth
            self.default_compression = CompressionType.ZLIB
            self.message_batch_size = 50
            self.transmission_timeout = 5.0
            
        elif platform == HardwarePlatform.JETSON_NANO:
            # Jetson Nano optimizations: Conservative resource usage
            self.default_compression = CompressionType.ZLIB
            self.message_batch_size = 20
            self.transmission_timeout = 10.0
            
        elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_WINDOWS]:
            # x86 optimizations: Higher throughput
            self.default_compression = CompressionType.ZLIB
            self.message_batch_size = 100
            self.transmission_timeout = 3.0
            
        else:
            # Default settings
            self.default_compression = CompressionType.ZLIB
            self.message_batch_size = 30
            self.transmission_timeout = 7.0
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def add_route(self, node_id: str, endpoint: NetworkEndpoint):
        """Add routing entry for a node"""
        self.routing_table[node_id] = endpoint
    
    def send_message(self, message: CrossPlatformMessage, target_endpoint: Optional[NetworkEndpoint] = None) -> bool:
        """Send message to target"""
        try:
            # Determine endpoint
            if target_endpoint is None:
                target_endpoint = self.routing_table.get(message.target_id)
                if not target_endpoint:
                    logging.error(f"No route to target: {message.target_id}")
                    return False
            
            # Serialize message
            message_data = json.dumps(message.to_dict()).encode('utf-8')
            
            # Compress if needed
            if target_endpoint.compression != CompressionType.NONE:
                compressed_data, compression_ratio = self.compression_manager.compress(
                    message_data, target_endpoint.compression
                )
                message.compression_ratio = compression_ratio
                message_data = compressed_data
            else:
                message.compression_ratio = 1.0
            
            # Send message
            start_time = time.time()
            success = self._transmit_data(message_data, target_endpoint, message)
            transmission_time = time.time() - start_time
            
            # Record metrics
            metrics = MessageMetrics(
                message_id=message.message_id,
                source_id=message.source_id,
                target_id=message.target_id,
                timestamp=message.timestamp,
                size_bytes=len(message_data),
                compression_ratio=message.compression_ratio,
                transmission_time=transmission_time,
                success=success,
                protocol=target_endpoint.protocol
            )
            self.metrics.append(metrics)
            
            return success
            
        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            return False
    
    def _transmit_data(self, data: bytes, endpoint: NetworkEndpoint, message: CrossPlatformMessage) -> bool:
        """Transmit data over network"""
        conn = self.connection_pool.get_connection(endpoint)
        if not conn:
            return False
        
        try:
            if endpoint.protocol == TransportProtocol.TCP:
                # Send message length first
                message_length = len(data).to_bytes(4, byteorder='big')
                conn.send(message_length)
                
                # Send message data
                conn.send(data)
                
                if message.requires_ack:
                    # Wait for acknowledgment
                    ack_data = conn.recv(4)
                    ack_length = int.from_bytes(ack_data, byteorder='big')
                    ack_data = conn.recv(ack_length)
                    ack = json.loads(ack_data.decode('utf-8'))
                    return ack.get('message_id') == message.message_id
                
                return True
                
            elif endpoint.protocol == TransportProtocol.UDP:
                # For UDP, send data directly (no guarantee)
                conn.sendto(data, (endpoint.host, endpoint.port))
                return not message.requires_ack  # Only success if no ACK required
            
            else:
                logging.error(f"Unsupported transport protocol: {endpoint.protocol}")
                return False
                
        except Exception as e:
            logging.error(f"Transmission failed: {e}")
            return False
        finally:
            self.connection_pool.release_connection(endpoint)
    
    def start_message_listener(self, endpoint: NetworkEndpoint):
        """Start listening for incoming messages"""
        def listener_worker():
            if endpoint.protocol == TransportProtocol.TCP:
                self._tcp_listener(endpoint)
            elif endpoint.protocol == TransportProtocol.UDP:
                self._udp_listener(endpoint)
        
        listener_thread = threading.Thread(target=listener_worker, daemon=True)
        listener_thread.start()
    
    def _tcp_listener(self, endpoint: NetworkEndpoint):
        """TCP message listener"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            if endpoint.ssl_enabled and endpoint.ssl_context:
                server_socket = endpoint.ssl_context.wrap_socket(server_socket, server_side=True)
            
            server_socket.bind((endpoint.host, endpoint.port))
            server_socket.listen(10)
            
            logging.info(f"TCP listener started on {endpoint.host}:{endpoint.port}")
            
            while True:
                try:
                    client_socket, addr = server_socket.accept()
                    
                    # Handle connection in separate thread
                    conn_thread = threading.Thread(
                        target=self._handle_tcp_connection,
                        args=(client_socket, addr, endpoint),
                        daemon=True
                    )
                    conn_thread.start()
                    
                except Exception as e:
                    logging.error(f"TCP listener error: {e}")
                    time.sleep(1.0)
                    
        except Exception as e:
            logging.error(f"Failed to start TCP listener: {e}")
    
    def _udp_listener(self, endpoint: NetworkEndpoint):
        """UDP message listener"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            server_socket.bind((endpoint.host, endpoint.port))
            
            logging.info(f"UDP listener started on {endpoint.host}:{endpoint.port}")
            
            while True:
                try:
                    data, addr = server_socket.recvfrom(65535)  # Max UDP packet size
                    
                    # Process message
                    self._process_received_data(data, endpoint)
                    
                except Exception as e:
                    logging.error(f"UDP listener error: {e}")
                    
        except Exception as e:
            logging.error(f"Failed to start UDP listener: {e}")
    
    def _handle_tcp_connection(self, client_socket: socket.socket, addr: Tuple[str, int], endpoint: NetworkEndpoint):
        """Handle individual TCP connection"""
        try:
            while True:
                # Receive message length
                length_data = client_socket.recv(4)
                if not length_data:
                    break
                
                message_length = int.from_bytes(length_data, byteorder='big')
                
                # Receive message data
                message_data = b''
                while len(message_data) < message_length:
                    chunk = client_socket.recv(min(message_length - len(message_data), 4096))
                    if not chunk:
                        break
                    message_data += chunk
                
                if len(message_data) == message_length:
                    # Process message
                    self._process_received_data(message_data, endpoint)
                    
                    # Send acknowledgment if needed
                    # (Implementation depends on message processing)
                    
        except Exception as e:
            logging.error(f"TCP connection error: {e}")
        finally:
            client_socket.close()
    
    def _process_received_data(self, data: bytes, endpoint: NetworkEndpoint):
        """Process received message data"""
        try:
            # Decompress if needed
            if endpoint.compression != CompressionType.NONE:
                decompressed_data = self.compression_manager.decompress(data, endpoint.compression)
            else:
                decompressed_data = data
            
            # Parse message
            message_dict = json.loads(decompressed_data.decode('utf-8'))
            message = CrossPlatformMessage.from_dict(message_dict)
            
            # Handle message
            self._handle_message(message)
            
        except Exception as e:
            logging.error(f"Failed to process received data: {e}")
    
    def _handle_message(self, message: CrossPlatformMessage):
        """Handle received message"""
        handlers = self.message_handlers.get(message.message_type, [])
        
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logging.error(f"Message handler error: {e}")
    
    def send_bundle(self, bundle: CommunicationBundle, target_endpoint: NetworkEndpoint) -> bool:
        """Send communication bundle"""
        try:
            # Create bundle message
            bundle_message = CrossPlatformMessage(
                message_id=f"bundle_{int(time.time() * 1000)}",
                source_id="router",
                target_id=target_endpoint.host,  # Simplified
                message_type="communication_bundle",
                payload={
                    'messages': bundle.messages,
                    'target_ids': bundle.target_ids,
                    'protocol': bundle.protocol.value,
                    'bundle_size': bundle.bundle_size
                },
                metadata={
                    'compression_ratio': bundle.compression_ratio,
                    'message_count': len(bundle.messages)
                }
            )
            
            return self.send_message(bundle_message, target_endpoint)
            
        except Exception as e:
            logging.error(f"Failed to send bundle: {e}")
            return False
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network communication statistics"""
        recent_metrics = [m for m in self.metrics if time.time() - m.timestamp < 300]  # Last 5 minutes
        
        if not recent_metrics:
            return {
                'total_messages': 0,
                'success_rate': 0.0,
                'avg_transmission_time': 0.0,
                'avg_compression_ratio': 1.0,
                'total_bytes_transmitted': 0
            }
        
        total_messages = len(recent_metrics)
        successful_messages = sum(1 for m in recent_metrics if m.success)
        total_bytes = sum(m.size_bytes for m in recent_metrics)
        
        return {
            'total_messages': total_messages,
            'success_rate': successful_messages / total_messages,
            'avg_transmission_time': sum(m.transmission_time for m in recent_metrics) / total_messages,
            'avg_compression_ratio': sum(m.compression_ratio for m in recent_metrics) / total_messages,
            'total_bytes_transmitted': total_bytes,
            'connection_pool_stats': self.connection_pool.get_stats()
        }

class ProtocolNegotiator:
    """Negotiates optimal communication protocols between platforms"""
    
    def __init__(self):
        self.protocol_preferences: Dict[HardwarePlatform, List[TransportProtocol]] = {
            HardwarePlatform.JETSON_ORIN: [TransportProtocol.TCP, TransportProtocol.UDP],
            HardwarePlatform.JETSON_NANO: [TransportProtocol.TCP, TransportProtocol.UDP],
            HardwarePlatform.X86_LINUX: [TransportProtocol.TCP, TransportProtocol.UDP, TransportProtocol.GRPC],
            HardwarePlatform.X86_WINDOWS: [TransportProtocol.TCP, TransportProtocol.UDP, TransportProtocol.WEBSOCKET],
            HardwarePlatform.X86_MACOS: [TransportProtocol.TCP, TransportProtocol.UDP],
            HardwarePlatform.ARM_LINUX: [TransportProtocol.TCP, TransportProtocol.UDP],
        }
    
    def negotiate_protocol(self, local_platform: HardwarePlatform, 
                          remote_platform: HardwarePlatform,
                          network_conditions: Dict[str, float]) -> TransportProtocol:
        """Negotiate optimal protocol between platforms"""
        
        # Get protocol preferences
        local_prefs = self.protocol_preferences.get(local_platform, [TransportProtocol.TCP])
        remote_prefs = self.protocol_preferences.get(remote_platform, [TransportProtocol.TCP])
        
        # Find common protocols
        common_protocols = list(set(local_prefs) & set(remote_prefs))
        if not common_protocols:
            return TransportProtocol.TCP  # Fallback
        
        # Select based on network conditions
        network_quality = network_conditions.get('quality', 0.5)
        latency = network_conditions.get('latency', 50.0)  # ms
        bandwidth = network_conditions.get('bandwidth', 100.0)  # Mbps
        
        if network_quality > 0.8 and latency < 10:
            # High quality, low latency - prefer TCP
            return TransportProtocol.TCP if TransportProtocol.TCP in common_protocols else common_protocols[0]
        elif bandwidth > 100:
            # High bandwidth - can use TCP overhead
            return TransportProtocol.TCP if TransportProtocol.TCP in common_protocols else common_protocols[0]
        else:
            # Poor conditions - use UDP for speed
            return TransportProtocol.UDP if TransportProtocol.UDP in common_protocols else common_protocols[0]
    
    def get_optimal_compression(self, network_conditions: Dict[str, float], 
                              message_size: int) -> CompressionType:
        """Get optimal compression based on conditions"""
        bandwidth = network_conditions.get('bandwidth', 100.0)  # Mbps
        cpu_load = network_conditions.get('cpu_load', 0.5)
        
        if message_size < 1024:  # Small messages - no compression
            return CompressionType.NONE
        elif bandwidth < 10 and cpu_load < 0.7:  # Low bandwidth, CPU available
            return CompressionType.ZLIB
        elif bandwidth < 50:  # Medium bandwidth
            return CompressionType.ZLIB
        else:  # High bandwidth
            return CompressionType.NONE

class PlatformBridge:
    """Bridges communication between different platforms"""
    
    def __init__(self, local_capabilities: HardwareCapabilities):
        self.local_capabilities = local_capabilities
        self.message_router = MessageRouter(local_capabilities)
        self.protocol_negotiator = ProtocolNegotiator()
        self.active_bridges: Dict[str, Dict[str, Any]] = {}
        
    def connect_to_platform(self, remote_node_id: str, 
                           remote_capabilities: HardwareCapabilities,
                           network_endpoint: NetworkEndpoint) -> bool:
        """Connect to remote platform"""
        try:
            # Negotiate optimal protocol
            network_conditions = self._detect_network_conditions(network_endpoint)
            optimal_protocol = self.protocol_negotiator.negotiate_protocol(
                self.local_capabilities.platform,
                remote_capabilities.platform,
                network_conditions
            )
            
            # Update endpoint protocol
            network_endpoint.protocol = optimal_protocol
            
            # Optimize compression
            compression = self.protocol_negotiator.get_optimal_compression(
                network_conditions, 1024  # Assume 1KB messages
            )
            network_endpoint.compression = compression
            
            # Add route
            self.message_router.add_route(remote_node_id, network_endpoint)
            
            # Record bridge
            self.active_bridges[remote_node_id] = {
                'capabilities': remote_capabilities,
                'endpoint': network_endpoint,
                'network_conditions': network_conditions,
                'connected_at': time.time()
            }
            
            logging.info(f"Connected to {remote_node_id} using {optimal_protocol.value}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to platform {remote_node_id}: {e}")
            return False
    
    def _detect_network_conditions(self, endpoint: NetworkEndpoint) -> Dict[str, float]:
        """Detect network conditions to endpoint"""
        try:
            # Simple ping test
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((endpoint.host, endpoint.port))
            sock.close()
            
            if result == 0:
                latency = (time.time() - start_time) * 1000  # Convert to ms
                quality = max(0.0, min(1.0, 100.0 / max(latency, 1)))  # Higher is better
                
                # Estimate bandwidth (simplified)
                bandwidth = 1000.0 if endpoint.host.startswith('192.168.') else 100.0
                
                return {
                    'quality': quality,
                    'latency': latency,
                    'bandwidth': bandwidth,
                    'cpu_load': psutil.cpu_percent() / 100.0
                }
            else:
                # Cannot connect
                return {
                    'quality': 0.0,
                    'latency': 9999.0,
                    'bandwidth': 0.1,
                    'cpu_load': psutil.cpu_percent() / 100.0
                }
                
        except Exception:
            return {
                'quality': 0.1,
                'latency': 1000.0,
                'bandwidth': 10.0,
                'cpu_load': psutil.cpu_percent() / 100.0
            }
    
    def send_cross_platform_message(self, message: CrossPlatformMessage) -> bool:
        """Send message to appropriate platform"""
        return self.message_router.send_message(message)
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get status of all active bridges"""
        return {
            'local_platform': self.local_capabilities.platform.value,
            'active_bridges': len(self.active_bridges),
            'bridge_details': {
                node_id: {
                    'platform': info['capabilities'].platform.value,
                    'protocol': info['endpoint'].protocol.value,
                    'compression': info['endpoint'].compression.value,
                    'connected_duration': time.time() - info['connected_at'],
                    'network_quality': info['network_conditions']['quality']
                }
                for node_id, info in self.active_bridges.items()
            },
            'network_stats': self.message_router.get_network_stats()
        }