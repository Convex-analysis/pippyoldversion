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
from collections import deque

from .hardware_adapter import HardwarePlatform, HardwareCapabilities, NetworkInterface
from .types import CommunicationBundle, CommunicationProtocol
import sys
sys.path.insert(0, '/Volumes/HardDriveMac/EXP/pippyoldversion')
try:
    from fhdp.vehicle_layer.communication import V2VMessage
except ImportError:
    # If V2VMessage is not available, create a placeholder for type hints
    V2VMessage = None

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
    ttl: float = 30.0
    requires_ack: bool = True
    priority: int = 0
    compression_type: CompressionType = CompressionType.ZLIB

    def _serialize_payload(self, obj: Any) -> Any:
        """递归序列化 payload，处理 Tensor 等不可 JSON 序列化的对象"""
        if hasattr(obj, 'tolist'):  # numpy array or torch tensor
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._serialize_payload(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_payload(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)  # 其他类型转字符串

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['compression_type'] = self.compression_type.value
        data['payload'] = self._serialize_payload(data['payload'])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrossPlatformMessage':
        # 必需字段检查：ACK 包或格式错误包会缺少这些字段，直接拒绝
        _required = {'message_id', 'source_id', 'target_id', 'message_type', 'payload'}
        _missing = _required - data.keys()
        if _missing:
            raise ValueError(
                f"CrossPlatformMessage.from_dict: missing required fields {_missing}; "
                f"received keys={set(data.keys())}"
            )
        if 'compression_type' in data and isinstance(data['compression_type'], str):
            data['compression_type'] = CompressionType(data['compression_type'])
        # 过滤掉 dataclass 不认识的额外字段，避免 unexpected keyword argument
        _known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in _known}
        return cls(**filtered)


class CompressionManager:
    """Manages compression algorithms"""

    def __init__(self):
        self.compression_map: dict[CompressionType, Callable[[bytes], bytes]] = {
            CompressionType.NONE: self._no_compress,
            CompressionType.ZLIB: self._zlib_compress,
        }
        self.decompression_map = {
            CompressionType.NONE: self._no_decompress,
            CompressionType.ZLIB: self._zlib_decompress,
        }

    def compress(self, data: bytes, compression_type: CompressionType) -> Tuple[bytes, float]:
        compress_func = self.compression_map.get(compression_type, self._no_compress)
        compressed = compress_func(data)
        compression_ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
        return compressed, compression_ratio

    def decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
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

    @staticmethod
    def is_zlib_compressed(data: bytes) -> bool:
        """Detect ZLIB magic bytes: 0x78 + {0x01, 0x5e, 0x9c, 0xda}"""
        return len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5e, 0x9c, 0xda)

    def auto_decompress(self, data: bytes) -> bytes:
        """Auto-detect and decompress if data is ZLIB compressed"""
        if self.is_zlib_compressed(data):
            try:
                return self._zlib_decompress(data)
            except Exception as e:
                logging.warning(f"Auto-decompression failed, using raw data: {e}")
        return data


class ConnectionPool:
    """Manages network connections efficiently"""

    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.active_connections: Dict[str, socket.socket] = {}
        self.connection_lock = threading.Lock()
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        # per-connection recv lock: prevents send_message ACK-reader and
        # start_receiving_from listener from racing on the same socket's recv
        self._recv_locks: Dict[str, threading.Lock] = {}
        # per-connection health check timestamp cache (5s)
        self._last_health_check: Dict[str, float] = {}

    def get_connection(self, endpoint: NetworkEndpoint) -> Optional[Tuple[socket.socket, threading.Lock]]:
        """Get or create connection; returns (socket, recv_lock) tuple, or None on failure."""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"

        with self.connection_lock:
            if connection_key in self.active_connections:
                conn = self.active_connections[connection_key]
                if self._is_connection_alive_cached(conn, connection_key):
                    return conn, self._recv_locks[connection_key]
                else:
                    try:
                        conn.close()
                    except:
                        pass
                    del self.active_connections[connection_key]
                    self._recv_locks.pop(connection_key, None)
                    self._last_health_check.pop(connection_key, None)

            if len(self.active_connections) >= self.max_connections:
                self._cleanup_connections()

            new_conn = self._create_connection(endpoint)
            if new_conn:
                self.active_connections[connection_key] = new_conn
                self._recv_locks[connection_key] = threading.Lock()
                self.connection_stats[connection_key] = {
                    'created': time.time(),
                    'last_used': time.time(),
                }
                self._last_health_check[connection_key] = time.time()
                return new_conn, self._recv_locks[connection_key]

            return None

    def _is_connection_alive_cached(self, conn: socket.socket, connection_key: str) -> bool:
        """
        健康检查带 5s 缓存：避免每次 get_connection 都触发 MSG_PEEK syscall
        """
        last_check = self._last_health_check.get(connection_key, 0)
        if time.time() - last_check < 5.0:
            return True  # 缓存期内跳过检查
        is_alive = self._is_connection_alive(conn)
        if is_alive:
            self._last_health_check[connection_key] = time.time()
        return is_alive

    def _is_connection_alive(self, conn: socket.socket) -> bool:
        """
        修复：原逻辑错误地把"有待读数据"判断为不健康。
        正确做法：用 MSG_PEEK 检查连接是否已被对端关闭（recv 返回空）。
        """
        try:
            # 检查 socket 文件描述符是否有效
            if conn.fileno() < 0:
                return False
            data = conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            if data == b'':
                # 对端已关闭连接
                return False
            # 有数据待读（如服务器推送），连接是健康的
            return True
        except BlockingIOError:
            # 无数据等待，连接正常
            return True
        except (OSError, ValueError):
            # 文件描述符已关闭或无效
            return False

    def _create_connection(self, endpoint: NetworkEndpoint) -> Optional[socket.socket]:
        """Create new connection with optimized socket parameters"""
        try:
            if endpoint.protocol == TransportProtocol.TCP:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(10.0)
                # Socket 参数调优
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 禁用 Nagle 算法
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)    # 启用保活
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144) # 256KB send buffer
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144) # 256KB recv buffer
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

    def invalidate_connection(self, endpoint: NetworkEndpoint):
        """Forcibly remove a connection from pool (e.g., after error)"""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"
        with self.connection_lock:
            if connection_key in self.active_connections:
                try:
                    self.active_connections[connection_key].close()
                except:
                    pass
                del self.active_connections[connection_key]
                self.connection_stats.pop(connection_key, None)
                self._recv_locks.pop(connection_key, None)
                self._last_health_check.pop(connection_key, None)

    def release_connection(self, endpoint: NetworkEndpoint):
        """Release connection back to pool (update last_used)"""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"
        with self.connection_lock:
            if connection_key in self.connection_stats:
                self.connection_stats[connection_key]['last_used'] = time.time()

    def _cleanup_connections(self):
        current_time = time.time()
        to_remove = [
            key for key, stats in self.connection_stats.items()
            if current_time - stats.get('last_used', 0) > 300
        ]
        for key in to_remove:
            try:
                self.active_connections[key].close()
            except:
                pass
            self.active_connections.pop(key, None)
            self.connection_stats.pop(key, None)
            self._recv_locks.pop(key, None)
            self._last_health_check.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
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
        self.metrics: deque[MessageMetrics] = deque(maxlen=10000)  # 环形缓冲防止内存泄漏
        self.compression_manager = CompressionManager()
        self.connection_pool = ConnectionPool()
        self._handler_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="msg_handler")  # 异步 handler 分发

        # server-side: stores sockets accepted from clients (for server→client push)
        self.client_connections: Dict[str, socket.socket] = {}
        # lock for client_connections
        self._client_conn_lock = threading.Lock()

        # client-side: incoming message listener thread control
        self._listener_threads: Dict[str, threading.Thread] = {}
        self._listener_running: Dict[str, bool] = {}

        self.optimize_for_platform()

    def optimize_for_platform(self):
        platform = self.hardware_capabilities.platform
        if platform == HardwarePlatform.JETSON_ORIN:
            self.default_compression = CompressionType.ZLIB
            self.message_batch_size = 50
            self.transmission_timeout = 5.0
        elif platform == HardwarePlatform.JETSON_NANO:
            self.default_compression = CompressionType.ZLIB
            self.message_batch_size = 20
            self.transmission_timeout = 10.0
        elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_WINDOWS]:
            self.default_compression = CompressionType.ZLIB
            self.message_batch_size = 100
            self.transmission_timeout = 3.0
        else:
            self.default_compression = CompressionType.ZLIB
            self.message_batch_size = 30
            self.transmission_timeout = 7.0

    def register_handler(self, message_type: str, handler: Callable[..., Any]):
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    def add_route(self, node_id: str, endpoint: NetworkEndpoint,
                  client_socket: Optional[socket.socket] = None):
        self.routing_table[node_id] = endpoint
        if client_socket:
            with self._client_conn_lock:
                self.client_connections[node_id] = client_socket

    def send_message(self, message: Union[CrossPlatformMessage, 'V2VMessage'],
                     target_endpoint: Optional[NetworkEndpoint] = None) -> bool:
        """发送消息，支持 CrossPlatformMessage 和 V2VMessage"""
        try:
            # 如果是 V2VMessage，转换为 CrossPlatformMessage
            if hasattr(message, 'sender_id'):  # V2VMessage
                target_id = message.receiver_id if hasattr(message, 'receiver_id') else None
                if target_id and target_endpoint is None:
                    target_endpoint = self.routing_table.get(target_id)
                    if not target_endpoint:
                        logging.error(f"No route to target: {target_id}")
                        return False

                # 转换 V2VMessage → CrossPlatformMessage
                cp_message = CrossPlatformMessage(
                    message_id=message.message_id,
                    source_id=message.sender_id,
                    target_id=message.receiver_id,
                    message_type=message.message_type,
                    payload=message.payload,
                    timestamp=message.timestamp,
                    requires_ack=True,
                    priority=0,
                    compression_type=CompressionType.ZLIB
                )
                return self._send_cross_platform_message(cp_message, target_endpoint)

            # CrossPlatformMessage 直接发送
            if target_endpoint is None:
                target_endpoint = self.routing_table.get(message.target_id)
                if not target_endpoint:
                    logging.error(f"No route to target: {message.target_id}")
                    return False

            return self._send_cross_platform_message(message, target_endpoint)

        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def _send_cross_platform_message(self, message: CrossPlatformMessage,
                                    target_endpoint: NetworkEndpoint) -> bool:
        """内部方法：发送 CrossPlatformMessage"""
        try:
            # Serialize
            message_data = json.dumps(message.to_dict()).encode('utf-8')

            # 修复：不修改原始 message 对象的 compression_type，使用局部变量
            compression_type = message.compression_type
            compression_ratio = 1.0
            if compression_type != CompressionType.NONE:
                message_data, compression_ratio = self.compression_manager.compress(
                    message_data, compression_type
                )

            start_time = time.time()
            success = self._transmit_data(message_data, target_endpoint, message)
            transmission_time = time.time() - start_time

            self.metrics.append(MessageMetrics(
                message_id=message.message_id,
                source_id=message.source_id,
                target_id=message.target_id,
                timestamp=message.timestamp,
                size_bytes=len(message_data),
                compression_ratio=compression_ratio,
                transmission_time=transmission_time,
                success=success,
                protocol=target_endpoint.protocol
            ))
            return success

        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            return False

    @staticmethod
    def _check_socket_alive(conn: socket.socket) -> bool:
        """检查 socket 是否可用（静态方法，供跨类使用）"""
        try:
            if conn.fileno() < 0:
                return False
            conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            return True
        except BlockingIOError:
            return True
        except (OSError, ValueError):
            return False

    def _transmit_data(self, data: bytes, endpoint: NetworkEndpoint,
                       message: CrossPlatformMessage) -> bool:
        """
        发送策略：
        1. 服务器→客户端：使用 client_connections 中的持久连接（无 ACK 等待）
        2. 客户端→服务器：使用 ConnectionPool（等待 ACK）
        """
        # --- 路径 1：服务器推送，使用 client_connections ---
        with self._client_conn_lock:
            client_conn = self.client_connections.get(message.target_id)

        if client_conn is not None:
            try:
                msg_len = len(data).to_bytes(4, byteorder='big')
                client_conn.sendall(msg_len + data)  # 合并为单次 sendall
                return True
            except Exception as e:
                logging.error(f"Failed to send via client connection to {message.target_id}: {e}")
                try:
                    client_conn.close()
                except:
                    pass
                with self._client_conn_lock:
                    self.client_connections.pop(message.target_id, None)
                return False

        # --- 路径 2：客户端发送，使用 ConnectionPool ---
        result = self.connection_pool.get_connection(endpoint)
        if result is None:
            return False
        conn, recv_lock = result

        try:
            if endpoint.protocol == TransportProtocol.TCP:
                msg_len = len(data).to_bytes(4, byteorder='big')
                conn.sendall(msg_len + data)  # 合并为单次 sendall

                if message.requires_ack:
                    # 读取 ACK（length-prefixed），全程持有 recv_lock 防止与接收线程竞争
                    with recv_lock:
                        ack_len_data = self._recv_exact(conn, 4)
                        if not ack_len_data:
                            logging.warning("Connection closed while waiting for ACK length")
                            self.connection_pool.invalidate_connection(endpoint)
                            return False
                        ack_length = int.from_bytes(ack_len_data, byteorder='big')
                        ack_data = self._recv_exact(conn, ack_length)
                        if not ack_data:
                            logging.warning("Connection closed while reading ACK data")
                            self.connection_pool.invalidate_connection(endpoint)
                            return False
                    # ACK 可能是压缩的，需要先解压
                    decompressed_ack = self.compression_manager.auto_decompress(ack_data)
                    ack = json.loads(decompressed_ack.decode('utf-8'))
                    return ack.get('message_id') == message.message_id

                return True

            elif endpoint.protocol == TransportProtocol.UDP:
                conn.sendto(data, (endpoint.host, endpoint.port))
                return not message.requires_ack

            else:
                logging.error(f"Unsupported transport protocol: {endpoint.protocol}")
                return False

        except OSError as e:
            # 处理 Bad file descriptor 等底层 socket 错误
            logging.error(f"Socket error during transmission: {e}")
            self.connection_pool.invalidate_connection(endpoint)
            return False
        except Exception as e:
            logging.error(f"Transmission failed: {e}")
            self.connection_pool.invalidate_connection(endpoint)
            return False
        finally:
            self.connection_pool.release_connection(endpoint)

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
        """
        优化的零拷贝接收：使用 bytearray + recv_into 预分配，
        避免频繁的 bytes 对象分配和拼接。
        """
        buf = bytearray(n)
        view = memoryview(buf)
        offset = 0
        while offset < n:
            received = conn.recv_into(view[offset:], n - offset)
            if received == 0:
                return None
            offset += received
        return bytes(buf)

    # ----------------------------------------------------------------
    # 服务器端：监听入站连接
    # ----------------------------------------------------------------

    def start_message_listener(self, endpoint: NetworkEndpoint):
        """Start listening for incoming messages (server mode)"""
        def listener_worker():
            if endpoint.protocol == TransportProtocol.TCP:
                self._tcp_listener(endpoint)
            elif endpoint.protocol == TransportProtocol.UDP:
                self._udp_listener(endpoint)

        t = threading.Thread(target=listener_worker, daemon=True)
        t.start()

    def _tcp_listener(self, endpoint: NetworkEndpoint):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Socket 参数调优（服务端）
            server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
            if endpoint.ssl_enabled and endpoint.ssl_context:
                server_socket = endpoint.ssl_context.wrap_socket(
                    server_socket, server_side=True)
            server_socket.bind((endpoint.host, endpoint.port))
            server_socket.listen(128)  # 提高 backlog 支持更多并发连接
            logging.info(f"TCP listener started on {endpoint.host}:{endpoint.port}")

            while True:
                try:
                    client_socket, addr = server_socket.accept()
                    # 新接受的连接也应用 socket 参数
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
                    t = threading.Thread(
                        target=self._handle_tcp_connection,
                        args=(client_socket, addr),
                        daemon=True
                    )
                    t.start()
                except Exception as e:
                    logging.error(f"TCP listener error: {e}")
                    time.sleep(1.0)
        except Exception as e:
            logging.error(f"Failed to start TCP listener: {e}")

    def _udp_listener(self, endpoint: NetworkEndpoint):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            server_socket.bind((endpoint.host, endpoint.port))
            logging.info(f"UDP listener started on {endpoint.host}:{endpoint.port}")
            while True:
                try:
                    data, addr = server_socket.recvfrom(65535)
                    decompressed = self.compression_manager.auto_decompress(data)
                    message_dict = json.loads(decompressed.decode('utf-8'))
                    self._handle_message(CrossPlatformMessage.from_dict(message_dict))
                except Exception as e:
                    logging.error(f"UDP listener error: {e}")
        except Exception as e:
            logging.error(f"Failed to start UDP listener: {e}")

    def _handle_tcp_connection(self, client_socket: socket.socket,
                                addr: Tuple[str, int]):
        """
        服务器端处理每个客户端连接的线程。
        - 自动将 source_id → client_socket 注册到 client_connections
        - 收到消息后立即发送 ACK（如果 requires_ack）
        - 收到 listen_register 消息时，将 client_connections 切换到该
          socket（专用推送连接），然后退出 recv 循环但**不关闭 socket**
        - 连接关闭时自动清理
        """
        source_node_id = None
        is_push_socket = False  # 标记：该 socket 是否已移交为推送专用
        try:
            client_socket.settimeout(300.0)

            while True:
                # 读取 4 字节长度前缀
                len_data = self._recv_exact(client_socket, 4)
                if not len_data:
                    break

                msg_len = int.from_bytes(len_data, byteorder='big')
                msg_data = self._recv_exact(client_socket, msg_len)
                if msg_data is None:
                    break

                # 自动检测并解压
                decompressed = self.compression_manager.auto_decompress(msg_data)

                # 解析消息
                message_dict = json.loads(decompressed.decode('utf-8'))
                try:
                    message = CrossPlatformMessage.from_dict(message_dict)
                except ValueError as ve:
                    logging.warning(
                        f"_handle_tcp_connection({addr}): discarding malformed packet: {ve}"
                    )
                    continue

                # --- listen_register：客户端专用接收 socket 的身份注册 ---
                if message.message_type == 'listen_register':
                    source_node_id = message.source_id
                    with self._client_conn_lock:
                        # 关闭旧的推送连接（如果有的话），切换到新的
                        old_conn = self.client_connections.get(source_node_id)
                        if old_conn is not None and old_conn is not client_socket:
                            # 不关闭旧连接——它可能是客户端的发送连接（ConnectionPool），
                            # 由 _handle_tcp_connection 的另一个线程管理
                            pass
                        self.client_connections[source_node_id] = client_socket
                    logging.info(
                        f"listen_register: switched push socket for "
                        f"{source_node_id} to {addr}")
                    is_push_socket = True
                    # 退出 recv 循环：这条连接现在专门用于服务器→客户端推送，
                    # 客户端不会再往上面发消息（避免双方互 recv 死锁）。
                    break

                # 注册 client_connections（仅首次，普通业务连接）
                source_node_id = message.source_id
                with self._client_conn_lock:
                    if source_node_id and source_node_id not in self.client_connections:
                        self.client_connections[source_node_id] = client_socket
                        logging.info(
                            f"Registered client connection for {source_node_id} from {addr}")

                # 发送 ACK
                if message.requires_ack:
                    ack_payload = json.dumps(
                        {'message_id': message.message_id}).encode('utf-8')
                    ack_len = len(ack_payload).to_bytes(4, byteorder='big')
                    try:
                        client_socket.sendall(ack_len + ack_payload)  # 合并为单次 sendall
                    except Exception as e:
                        logging.warning(f"Failed to send ACK to {source_node_id}: {e}")

                # 分发消息
                try:
                    self._handle_message(message)
                except Exception as e:
                    logging.error(f"Error handling message from {addr}: {e}")

        except Exception as e:
            logging.error(f"TCP connection error from {addr}: {e}")
        finally:
            if is_push_socket:
                # 推送专用 socket：不关闭、不清理 client_connections，
                # 由 _transmit_data 路径 1 继续使用
                logging.info(
                    f"Push socket handler exiting for {source_node_id}, "
                    f"socket kept alive for server push")
            else:
                with self._client_conn_lock:
                    if source_node_id and source_node_id in self.client_connections:
                        # 只有当 client_connections 里存的还是这个 socket 时才清理
                        # （可能已被 listen_register 替换为专用 socket）
                        if self.client_connections[source_node_id] is client_socket:
                            del self.client_connections[source_node_id]
                            logging.info(f"Removed client connection for {source_node_id}")
                try:
                    client_socket.close()
                except:
                    pass

    # ----------------------------------------------------------------
    # 客户端端：主动监听服务器推送（新增）
    # ----------------------------------------------------------------

    def start_receiving_from(self, node_id: str,
                             self_node_id: Optional[str] = None):
        """
        客户端调用此方法，在后台持续接收来自 node_id 的推送消息。

        使用**独立专用 socket**（不走 ConnectionPool），与发送路径完全隔离：
        - 发送线程（ConnectionPool）和接收线程不再共用同一个 socket
        - 避免两个线程竞争 settimeout / recv，消除 Bad file descriptor 问题

        连接建立后先发送 listen_register 消息，让服务器将 client_connections
        切换到这条专用连接，后续推送全部走该 socket。

        用法（客户端）：
            router.add_route("server", server_endpoint)
            router.start_receiving_from("server", self_node_id="agx_orin_001")
        """
        if node_id in self._listener_running and self._listener_running[node_id]:
            return  # 已经在监听

        self._listener_running[node_id] = True

        def _make_dedicated_socket(endpoint: NetworkEndpoint) -> Optional[socket.socket]:
            """建立专用接收 socket，不走 ConnectionPool"""
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
                conn.settimeout(30.0)
                conn.connect((endpoint.host, endpoint.port))
                return conn
            except Exception as e:
                logging.error(f"start_receiving_from: dedicated socket connect failed: {e}")
                return None

        def _send_listen_register(conn: socket.socket) -> bool:
            """发送 listen_register 消息，让服务器识别该连接属于本节点"""
            if not self_node_id:
                return True  # 未提供 node_id 则跳过
            try:
                reg_msg = json.dumps({
                    'message_id': f'listen_reg_{self_node_id}',
                    'source_id': self_node_id,
                    'target_id': node_id,
                    'message_type': 'listen_register',
                    'payload': {},
                    'timestamp': time.time(),
                    'requires_ack': False,
                    'priority': 0,
                    'compression_type': 'none'
                }).encode('utf-8')
                # 可能需要压缩（与服务器协议一致使用 zlib）
                compressed = self.compression_manager.compress(
                    reg_msg, CompressionType.ZLIB)[0]
                msg_len = len(compressed).to_bytes(4, byteorder='big')
                conn.sendall(msg_len + compressed)
                logging.info(f"Sent listen_register for {self_node_id}")
                return True
            except Exception as e:
                logging.error(f"Failed to send listen_register: {e}")
                return False

        def _run():
            endpoint = self.routing_table.get(node_id)
            if not endpoint:
                logging.error(f"start_receiving_from: no route to {node_id}")
                return

            while self._listener_running.get(node_id, False):
                conn = _make_dedicated_socket(endpoint)
                if conn is None:
                    logging.error(f"start_receiving_from: cannot connect to {node_id}, retry in 3s")
                    time.sleep(3.0)
                    continue

                # 发送身份标识，让服务器将推送连接切换到这个 socket
                if not _send_listen_register(conn):
                    try:
                        conn.close()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    continue

                try:
                    while self._listener_running.get(node_id, False):
                        try:
                            len_data = self._recv_exact(conn, 4)
                            if not len_data:
                                logging.info(
                                    f"Server {node_id} closed connection, reconnecting...")
                                break

                            msg_len = int.from_bytes(len_data, byteorder='big')
                            msg_data = self._recv_exact(conn, msg_len)
                            if msg_data is None:
                                break

                            decompressed = self.compression_manager.auto_decompress(msg_data)
                            message_dict = json.loads(decompressed.decode('utf-8'))

                            # 捕获 ACK 包误入（缺少必需字段）：丢弃并继续，不断连
                            try:
                                message = CrossPlatformMessage.from_dict(message_dict)
                            except ValueError as ve:
                                logging.warning(
                                    f"start_receiving_from({node_id}): discarding "
                                    f"non-business packet: {ve}"
                                )
                                continue

                            # 不回 ACK：服务器推送走 client_connections 路径，
                            # 不等 ACK，客户端回 ACK 会被 _handle_tcp_connection
                            # 的下次 recv 当作业务消息读入，导致 JSON 解析崩溃。

                            # 分发消息
                            try:
                                self._handle_message(message)
                            except Exception as e:
                                logging.error(f"Error handling message from {node_id}: {e}")

                        except socket.timeout:
                            continue  # 正常超时，继续等待
                        except Exception as e:
                            logging.error(f"Receive error from {node_id}: {e}")
                            break  # 断开重连

                except Exception as e:
                    logging.error(f"Listener connection error to {node_id}: {e}")
                finally:
                    # 只关闭专用 socket，不影响 ConnectionPool 中的发送连接
                    try:
                        conn.close()
                    except Exception:
                        pass
                    time.sleep(1.0)

        t = threading.Thread(target=_run, daemon=True,
                             name=f"recv-from-{node_id}")
        self._listener_threads[node_id] = t
        t.start()
        logging.info(f"Started receiving listener for {node_id}")

    def stop_receiving_from(self, node_id: str):
        """Stop the client-side listener for a specific node"""
        self._listener_running[node_id] = False

    # ----------------------------------------------------------------

    def _handle_message(self, message: CrossPlatformMessage):
        """异步分发消息到 handler，避免阻塞接收线程"""
        handlers = self.message_handlers.get(message.message_type, [])
        for handler in handlers:
            self._handler_executor.submit(self._run_handler, handler, message)

    def _run_handler(self, handler: Callable[..., Any], message: CrossPlatformMessage):
        """在独立线程中执行 handler"""
        try:
            handler(message)
        except Exception as e:
            logging.error(f"Message handler error: {e}")

    def send_bundle(self, bundle: CommunicationBundle,
                    target_endpoint: NetworkEndpoint) -> bool:
        try:
            bundle_message = CrossPlatformMessage(
                message_id=f"bundle_{int(time.time() * 1000)}",
                source_id="router",
                target_id=target_endpoint.host,
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
        current_time = time.time()
        recent_metrics = [m for m in self.metrics if current_time - m.timestamp < 300]
        if not recent_metrics:
            return {
                'total_messages': 0,
                'success_rate': 0.0,
                'avg_transmission_time': 0.0,
                'avg_compression_ratio': 1.0,
                'total_bytes_transmitted': 0
            }
        total = len(recent_metrics)
        return {
            'total_messages': total,
            'success_rate': sum(1 for m in recent_metrics if m.success) / total,
            'avg_transmission_time': sum(m.transmission_time for m in recent_metrics) / total,
            'avg_compression_ratio': sum(m.compression_ratio for m in recent_metrics) / total,
            'total_bytes_transmitted': sum(m.size_bytes for m in recent_metrics),
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
        local_prefs = self.protocol_preferences.get(local_platform, [TransportProtocol.TCP])
        remote_prefs = self.protocol_preferences.get(remote_platform, [TransportProtocol.TCP])
        common = list(set(local_prefs) & set(remote_prefs))
        if not common:
            return TransportProtocol.TCP
        latency = network_conditions.get('latency', 50.0)
        bandwidth = network_conditions.get('bandwidth', 100.0)
        quality = network_conditions.get('quality', 0.5)
        if quality > 0.8 and latency < 10:
            return TransportProtocol.TCP if TransportProtocol.TCP in common else common[0]
        elif bandwidth > 100:
            return TransportProtocol.TCP if TransportProtocol.TCP in common else common[0]
        else:
            return TransportProtocol.UDP if TransportProtocol.UDP in common else common[0]

    def get_optimal_compression(self, network_conditions: Dict[str, float],
                                 message_size: int) -> CompressionType:
        """
        修复：不再因高带宽而关掉压缩。
        保持 ZLIB 作为默认，仅对极小消息关闭。
        """
        if message_size < 256:
            return CompressionType.NONE
        return CompressionType.ZLIB


class PlatformBridge:
    """Bridges communication between different platforms"""

    def __init__(self, local_capabilities: HardwareCapabilities,
                 node_id: Optional[str] = None):
        self.local_capabilities = local_capabilities
        self.node_id = node_id  # 本节点 ID，用于接收线程向服务器注册身份
        self.message_router = MessageRouter(local_capabilities)
        self.protocol_negotiator = ProtocolNegotiator()
        self.active_bridges: Dict[str, Dict[str, Any]] = {}

    def connect_to_platform(self, remote_node_id: str,
                            remote_capabilities: HardwareCapabilities,
                            network_endpoint: NetworkEndpoint,
                            start_receiving: bool = True) -> bool:
        """
        Connect to remote platform.
        start_receiving=True（默认）：自动启动后台接收线程，
        客户端调用时无需额外调用 start_receiving_from()。
        """
        try:
            network_conditions = self._detect_network_conditions(network_endpoint)
            optimal_protocol = self.protocol_negotiator.negotiate_protocol(
                self.local_capabilities.platform,
                remote_capabilities.platform,
                network_conditions
            )
            network_endpoint.protocol = optimal_protocol

            # 修复：压缩策略不覆盖 endpoint，保持 ZLIB
            # （get_optimal_compression 已修复，但此处显式保持 ZLIB 更安全）
            network_endpoint.compression = CompressionType.ZLIB

            self.message_router.add_route(remote_node_id, network_endpoint)

            # 自动启动接收线程（客户端场景）
            if start_receiving:
                self.message_router.start_receiving_from(
                    remote_node_id, self_node_id=self.node_id)

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
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((endpoint.host, endpoint.port))
            sock.close()

            if result == 0:
                latency = (time.time() - start_time) * 1000
                quality = max(0.0, min(1.0, 100.0 / max(latency, 1)))
                bandwidth = 1000.0 if endpoint.host.startswith('192.168.') else 100.0
                return {
                    'quality': quality,
                    'latency': latency,
                    'bandwidth': bandwidth,
                    'cpu_load': psutil.cpu_percent() / 100.0
                }
            else:
                return {'quality': 0.0, 'latency': 9999.0,
                        'bandwidth': 0.1, 'cpu_load': psutil.cpu_percent() / 100.0}
        except Exception:
            return {'quality': 0.1, 'latency': 1000.0,
                    'bandwidth': 10.0, 'cpu_load': psutil.cpu_percent() / 100.0}

    def send_cross_platform_message(self, message: CrossPlatformMessage) -> bool:
        return self.message_router.send_message(message)

    def get_bridge_status(self) -> Dict[str, Any]:
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
