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

# Try to import psutil for system resource monitoring
psutil_available = False
try:
    import psutil
    psutil_available = True
except ImportError:
    pass

# Try to import faster JSON libraries
faster_json_available = False
try:
    import orjson
    faster_json_available = True
except ImportError:
    try:
        import ujson
        faster_json_available = True
    except ImportError:
        pass

# Try to import additional compression libraries
try:
    import lz4.frame
    lz4_available = True
except ImportError:
    lz4_available = False
    
try:
    import brotli
    brotli_available = True
except ImportError:
    brotli_available = False
    
try:
    import snappy
    snappy_available = True
except ImportError:
    snappy_available = False

# Try to import binary serialization libraries
binary_serialization_available = False
try:
    import msgpack
    binary_serialization_available = True
except ImportError:
    try:
        import protobuf
        binary_serialization_available = True
    except ImportError:
        pass
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

class SerializationFormat(Enum):
    """Serialization formats"""
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"

class SerializationManager:
    """Manages serialization/deserialization with automatic backend selection"""
    
    def __init__(self, default_format: SerializationFormat = SerializationFormat.JSON):
        self.default_format = default_format
        self._setup_serializers()
    
    def _setup_serializers(self):
        # Setup JSON serializers
        self.json_dumps = json.dumps
        self.json_loads = json.loads
        
        if faster_json_available:
            if 'orjson' in globals():
                self.json_dumps = lambda obj: orjson.dumps(obj, default=str).decode('utf-8')
                self.json_loads = lambda data: orjson.loads(data)
            elif 'ujson' in globals():
                self.json_dumps = ujson.dumps
                self.json_loads = ujson.loads
        
        # Setup binary serializers
        self.binary_available = False
        self.binary_dumps = None
        self.binary_loads = None
        
        if binary_serialization_available:
            if 'msgpack' in globals():
                self.binary_dumps = msgpack.packb
                self.binary_loads = msgpack.unpackb
                self.binary_available = True
    
    def serialize(self, data: Any, format: Optional[SerializationFormat] = None) -> bytes:
        """Serialize data using the specified format"""
        format = format or self.default_format
        
        if format == SerializationFormat.JSON:
            return self.json_dumps(data).encode('utf-8')
        elif format == SerializationFormat.MSGPACK and self.binary_available:
            return self.binary_dumps(data, default=str)
        else:
            # Fallback to JSON
            return self.json_dumps(data).encode('utf-8')
    
    def deserialize(self, data: bytes, format: Optional[SerializationFormat] = None) -> Any:
        """Deserialize data using the specified format"""
        format = format or self.default_format
        
        if format == SerializationFormat.JSON:
            return self.json_loads(data)
        elif format == SerializationFormat.MSGPACK and self.binary_available:
            return self.binary_loads(data)
        else:
            # Try to detect format
            try:
                # First try JSON
                return self.json_loads(data)
            except:
                # Then try msgpack if available
                if self.binary_available:
                    try:
                        return self.binary_loads(data)
                    except:
                        pass
                # Last resort: return as string
                return data.decode('utf-8', errors='ignore')

@dataclass
class NetworkEndpoint:
    """Network endpoint configuration"""
    host: str
    port: int
    protocol: TransportProtocol
    ssl_enabled: bool = False
    ssl_context: Optional[ssl.SSLContext] = None
    compression: CompressionType = CompressionType.ZLIB
    serialization_format: SerializationFormat = SerializationFormat.JSON

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
    
    # Pipeline-specific metrics
    pipeline_id: Optional[str] = None
    stage_id: Optional[str] = None
    sequence_id: Optional[str] = None
    sequence_index: int = 0
    data_type: Optional[str] = None
    
    # Performance metrics
    serialization_time: float = 0.0
    compression_time: float = 0.0
    deserialization_time: float = 0.0
    decompression_time: float = 0.0
    queue_time: float = 0.0
    processing_time: float = 0.0
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # Compression details
    compression_algorithm: Optional[str] = None
    serialization_format: Optional[str] = None
    
    # Connection details
    connection_type: str = "regular"  # regular, pipeline, preheated
    connection_reused: bool = False

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
    serialization_format: SerializationFormat = SerializationFormat.JSON

    def _serialize_payload(self, obj: Any) -> Any:
        """递归序列化 payload，处理 Tensor 等不可 JSON 序列化的对象"""
        try:
            if hasattr(obj, 'tolist'):  # numpy array or torch tensor
                # 处理大型数组的安全转换
                import numpy as np
                # Higher threshold for training-related data to support model parameters
                # Regular messages: 10,000 elements limit
                # Training data: 1,000,000 elements limit to support large model parameters
                max_elements = 1000000 if hasattr(self, 'message_type') and self.message_type in ['global_model', 'model_update', 'pipeline_data'] else 10000
                if hasattr(obj, 'shape') and np.prod(obj.shape) > max_elements:
                    return f"<Array shape={obj.shape} dtype={obj.dtype}>"  # 返回元数据而非实际数据
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: self._serialize_payload(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                # Keep recursive processing for lists
                return [self._serialize_payload(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            elif hasattr(obj, '__dict__'):  # 处理自定义对象
                return {k: self._serialize_payload(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            else:
                return str(obj)  # 其他类型转字符串
        except Exception as e:
            logging.warning(f"Failed to serialize object {type(obj).__name__}: {e}, using string representation")
            return str(obj)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['compression_type'] = self.compression_type.value
        data['serialization_format'] = self.serialization_format.value
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
        if 'serialization_format' in data and isinstance(data['serialization_format'], str):
            data['serialization_format'] = SerializationFormat(data['serialization_format'])
        # 过滤掉 dataclass 不认识的额外字段，避免 unexpected keyword argument
        _known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in _known}
        return cls(**filtered)

@dataclass
class BatchMessage:
    """Batch message container for efficient bulk transmission"""
    batch_id: str
    source_id: str
    target_id: str
    messages: List[CrossPlatformMessage]
    timestamp: float = field(default_factory=time.time)
    batch_size: int = field(default=0)
    compression_type: CompressionType = CompressionType.ZLIB
    serialization_format: SerializationFormat = SerializationFormat.JSON
    
    def __post_init__(self):
        if self.batch_size == 0:
            self.batch_size = len(self.messages)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['compression_type'] = self.compression_type.value
        data['serialization_format'] = self.serialization_format.value
        data['messages'] = [msg.to_dict() for msg in self.messages]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchMessage':
        # Convert messages back to CrossPlatformMessage objects
        messages = [CrossPlatformMessage.from_dict(msg_dict) for msg_dict in data['messages']]
        
        # Convert enums back
        if isinstance(data['compression_type'], str):
            data['compression_type'] = CompressionType(data['compression_type'])
        if isinstance(data['serialization_format'], str):
            data['serialization_format'] = SerializationFormat(data['serialization_format'])
            
        # Create batch message
        return cls(
            batch_id=data['batch_id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            messages=messages,
            timestamp=data['timestamp'],
            batch_size=data['batch_size'],
            compression_type=data['compression_type'],
            serialization_format=data['serialization_format']
        )

@dataclass
class PipelineMessage:
    """Pipeline-specific message for efficient inter-stage communication"""
    pipeline_id: str
    stage_id: str
    source_id: str
    target_id: str
    data: Any  # Pipeline data (tensors, activations, gradients, etc.)
    data_type: str  # Type of data (input, activation, gradient, checkpoint, etc.)
    sequence_id: str  # Sequence identifier for ordered processing
    sequence_index: int = 0  # Position in sequence
    total_sequence_length: int = 1  # Total number of messages in sequence
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    requires_ack: bool = True
    compression_type: CompressionType = CompressionType.ZLIB
    serialization_format: SerializationFormat = SerializationFormat.JSON
    
    def __post_init__(self):
        # Add pipeline-specific metadata
        self.metadata.update({
            'pipeline_id': self.pipeline_id,
            'stage_id': self.stage_id,
            'sequence_id': self.sequence_id,
            'sequence_index': self.sequence_index,
            'data_type': self.data_type
        })
    
    def _serialize_data(self, obj: Any) -> Any:
        """Specialized serialization for pipeline data"""
        try:
            if hasattr(obj, 'tolist'):  # numpy array or torch tensor
                # Pipeline data often contains large tensors, so we need efficient serialization
                import numpy as np
                if hasattr(obj, 'shape') and np.prod(obj.shape) > 50000:  # 更大的限制，流水线数据需要更多细节
                    # For extremely large tensors, we might want to split them
                    return {
                        '__tensor_metadata__': True,
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'data': obj.tolist()  # Still convert to list for now
                    }
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: self._serialize_data(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._serialize_data(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            elif hasattr(obj, '__dict__'):  # 处理自定义对象
                return {k: self._serialize_data(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            else:
                return str(obj)  # 其他类型转字符串
        except Exception as e:
            logging.warning(f"Failed to serialize pipeline data {type(obj).__name__}: {e}, using string representation")
            return str(obj)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['compression_type'] = self.compression_type.value
        data['serialization_format'] = self.serialization_format.value
        data['data'] = self._serialize_data(data['data'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineMessage':
        # Convert enums back
        if isinstance(data['compression_type'], str):
            data['compression_type'] = CompressionType(data['compression_type'])
        if isinstance(data['serialization_format'], str):
            data['serialization_format'] = SerializationFormat(data['serialization_format'])
        
        # Create pipeline message
        return cls(**data)

class PipelineCommunicationManager:
    """Manager for pipeline-specific communication patterns"""
    
    def __init__(self, message_router: 'MessageRouter'):
        self.message_router = message_router
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        self.sequence_buffers: Dict[str, Dict[int, PipelineMessage]] = {}  # sequence_id -> {index: message}
        self.sequence_handlers: Dict[str, Callable] = {}  # sequence_id -> handler
    
    def register_pipeline(self, pipeline_id: str, stages: List[str], network_endpoints: Dict[str, NetworkEndpoint]):
        """Register a new pipeline with its stages and endpoints"""
        self.active_pipelines[pipeline_id] = {
            'stages': stages,
            'endpoints': network_endpoints,
            'created_at': time.time()
        }
    
    def send_pipeline_data(self, pipeline_message: PipelineMessage, target_endpoint: NetworkEndpoint) -> bool:
        """Send pipeline data to the next stage"""
        try:
            # Convert to CrossPlatformMessage for transmission
            cp_message = CrossPlatformMessage(
                message_id=f"pipeline_{pipeline_message.pipeline_id}_{pipeline_message.sequence_id}_{pipeline_message.sequence_index}",
                source_id=pipeline_message.source_id,
                target_id=pipeline_message.target_id,
                message_type="pipeline_data",
                payload=pipeline_message.to_dict(),
                metadata=pipeline_message.metadata,
                requires_ack=pipeline_message.requires_ack,
                compression_type=pipeline_message.compression_type,
                serialization_format=pipeline_message.serialization_format
            )
            
            # Send using pipeline-specific connection for better performance
            return self._send_pipeline_message(cp_message, target_endpoint, pipeline_message.pipeline_id)
        except Exception as e:
            logging.error(f"Failed to send pipeline data: {e}")
            return False
    
    def send_pipeline_batch(self, pipeline_messages: List[PipelineMessage], target_endpoint: NetworkEndpoint) -> bool:
        """Send multiple pipeline messages in a batch"""
        if not pipeline_messages:
            return True
            
        try:
            # Convert to CrossPlatformMessage for batch transmission
            cp_messages = []
            for msg in pipeline_messages:
                cp_msg = CrossPlatformMessage(
                    message_id=f"pipeline_{msg.pipeline_id}_{msg.sequence_id}_{msg.sequence_index}",
                    source_id=msg.source_id,
                    target_id=msg.target_id,
                    message_type="pipeline_data",
                    payload=msg.to_dict(),
                    metadata=msg.metadata,
                    requires_ack=msg.requires_ack,
                    compression_type=msg.compression_type,
                    serialization_format=msg.serialization_format
                )
                cp_messages.append(cp_msg)
            
            # Get pipeline ID from first message
            pipeline_id = pipeline_messages[0].pipeline_id if pipeline_messages else None
            
            # Send using pipeline-specific connection for better performance
            return self._send_pipeline_batch(cp_messages, target_endpoint, pipeline_id)
        except Exception as e:
            logging.error(f"Failed to send pipeline batch: {e}")
            return False
            
    def _send_pipeline_message(self, cp_message: CrossPlatformMessage, target_endpoint: NetworkEndpoint, pipeline_id: str) -> bool:
        """Internal method to send pipeline message using pipeline-specific connection"""
        try:
            # Get pipeline connection
            result = self.message_router.connection_pool.get_pipeline_connection(pipeline_id, target_endpoint)
            if result is None:
                # Fallback to regular connection if pipeline connection fails
                return self.message_router.send_message(cp_message, target_endpoint)
                
            conn, recv_lock = result
            
            # Serialize message
            message_dict = cp_message.to_dict()
            serialized_data = self.message_router.serialization_manager.serialize(message_dict, cp_message.serialization_format)
            
            # Compress if needed
            if cp_message.compression_type != CompressionType.NONE:
                serialized_data, compression_ratio = self.message_router.compression_manager.compress(
                    serialized_data, cp_message.compression_type
                )
            
            # Send message
            msg_len = len(serialized_data).to_bytes(4, byteorder='big')
            conn.sendall(msg_len + serialized_data)
            
            # Send ACK if requested
            if cp_message.requires_ack:
                with recv_lock:
                    ack_len_data = self.message_router._recv_exact(conn, 4)
                    if not ack_len_data:
                        return False
                    ack_length = int.from_bytes(ack_len_data, byteorder='big')
                    ack_data = self.message_router._recv_exact(conn, ack_length)
                    if not ack_data:
                        return False
            
            # Update connection usage stats
            self.message_router.connection_pool.release_connection(target_endpoint, pipeline_id=pipeline_id)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send pipeline message: {e}")
            self.message_router.connection_pool.invalidate_connection(target_endpoint, pipeline_id=pipeline_id)
            # Fallback to regular connection
            return self.message_router.send_message(cp_message, target_endpoint)
            
    def _send_pipeline_batch(self, cp_messages: List[CrossPlatformMessage], target_endpoint: NetworkEndpoint, pipeline_id: Optional[str]) -> bool:
        """Internal method to send pipeline batch using pipeline-specific connection"""
        if not cp_messages:
            return True
            
        try:
            # Get pipeline connection if available
            result = None
            if pipeline_id:
                result = self.message_router.connection_pool.get_pipeline_connection(pipeline_id, target_endpoint)
                
            if result is None:
                # Fallback to regular batch sending
                return self.message_router.send_batch(cp_messages, target_endpoint)
                
            conn, recv_lock = result
            
            # Use the first message's compression type and serialization format
            compression_type = cp_messages[0].compression_type
            serialization_format = cp_messages[0].serialization_format
            
            # Create batch message
            batch_id = f"batch_{int(time.time() * 1000)}_{id(cp_messages)}"
            batch_source_id = cp_messages[0].source_id
            batch_target_id = cp_messages[0].target_id
            
            batch_message = BatchMessage(
                batch_id=batch_id,
                source_id=batch_source_id,
                target_id=batch_target_id,
                messages=cp_messages,
                compression_type=compression_type,
                serialization_format=serialization_format
            )
            
            # Serialize and send batch
            batch_dict = batch_message.to_dict()
            batch_data = self.message_router.serialization_manager.serialize(batch_dict, serialization_format)
            
            # Compress if needed
            if compression_type != CompressionType.NONE:
                batch_data, compression_ratio = self.message_router.compression_manager.compress(batch_data, compression_type)
            
            # Send batch
            msg_len = len(batch_data).to_bytes(4, byteorder='big')
            conn.sendall(msg_len + batch_data)
            
            # Send ACK if requested
            if any(msg.requires_ack for msg in cp_messages):
                with recv_lock:
                    ack_len_data = self.message_router._recv_exact(conn, 4)
                    if not ack_len_data:
                        return False
                    ack_length = int.from_bytes(ack_len_data, byteorder='big')
                    ack_data = self.message_router._recv_exact(conn, ack_length)
                    if not ack_data:
                        return False
            
            # Update connection usage stats
            if pipeline_id:
                self.message_router.connection_pool.release_connection(target_endpoint, pipeline_id=pipeline_id)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send pipeline batch: {e}")
            if pipeline_id:
                self.message_router.connection_pool.invalidate_connection(target_endpoint, pipeline_id=pipeline_id)
            # Fallback to regular batch sending
            return self.message_router.send_batch(cp_messages, target_endpoint)
    
    def handle_pipeline_message(self, message: CrossPlatformMessage):
        """Handle incoming pipeline messages"""
        if message.message_type != "pipeline_data":
            return
            
        try:
            # Convert back to PipelineMessage
            pipeline_msg = PipelineMessage.from_dict(message.payload)
            
            # Check if this is part of a sequence
            sequence_id = pipeline_msg.sequence_id
            
            if sequence_id not in self.sequence_buffers:
                self.sequence_buffers[sequence_id] = {}
            
            # Store the message
            self.sequence_buffers[sequence_id][pipeline_msg.sequence_index] = pipeline_msg
            
            # Check if we have all messages in the sequence
            buffer = self.sequence_buffers[sequence_id]
            expected_count = pipeline_msg.total_sequence_length
            
            if len(buffer) == expected_count:
                # Process the complete sequence
                ordered_messages = [buffer[i] for i in range(expected_count)]
                
                # Call the registered handler if any
                if sequence_id in self.sequence_handlers:
                    try:
                        self.sequence_handlers[sequence_id](ordered_messages)
                    except Exception as e:
                        logging.error(f"Failed to process pipeline sequence {sequence_id}: {e}")
                
                # Clean up
                del self.sequence_buffers[sequence_id]
        except Exception as e:
            logging.error(f"Failed to handle pipeline message: {e}")
    
    def register_sequence_handler(self, sequence_id: str, handler: Callable[[List[PipelineMessage]], None]):
        """Register a handler for a specific pipeline sequence"""
        self.sequence_handlers[sequence_id] = handler
    
    def unregister_sequence_handler(self, sequence_id: str):
        """Unregister a sequence handler"""
        if sequence_id in self.sequence_handlers:
            del self.sequence_handlers[sequence_id]


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
        
        # Add support for additional compression algorithms if available
        if lz4_available:
            self.compression_map[CompressionType.LZ4] = self._lz4_compress
            self.decompression_map[CompressionType.LZ4] = self._lz4_decompress
        
        if brotli_available:
            self.compression_map[CompressionType.BROTLI] = self._brotli_compress
            self.decompression_map[CompressionType.BROTLI] = self._brotli_decompress
        
        # Compression algorithm properties for intelligent selection
        self.compression_properties = {
            CompressionType.NONE: {
                'speed': 100,
                'compression_ratio': 1.0,
                'cpu_usage': 0,
                'memory_usage': 0
            },
            CompressionType.ZLIB: {
                'speed': 70,
                'compression_ratio': 0.4,
                'cpu_usage': 50,
                'memory_usage': 30
            },
        }
        
        if lz4_available:
            self.compression_properties[CompressionType.LZ4] = {
                'speed': 95,
                'compression_ratio': 0.6,
                'cpu_usage': 20,
                'memory_usage': 10
            }
        
        if brotli_available:
            self.compression_properties[CompressionType.BROTLI] = {
                'speed': 40,
                'compression_ratio': 0.3,
                'cpu_usage': 80,
                'memory_usage': 50
            }
        
        # Data type to optimal compression algorithm mapping
        self.data_type_mapping = {
            'pipeline_data_input': CompressionType.LZ4 if lz4_available else CompressionType.ZLIB,
            'pipeline_data_activation': CompressionType.LZ4 if lz4_available else CompressionType.ZLIB,
            'pipeline_data_gradient': CompressionType.BROTLI if brotli_available else CompressionType.ZLIB,
            'pipeline_data_checkpoint': CompressionType.BROTLI if brotli_available else CompressionType.ZLIB,
            'regular_message': CompressionType.ZLIB,
            'batch_message': CompressionType.LZ4 if lz4_available else CompressionType.ZLIB,
            'large_data': CompressionType.BROTLI if brotli_available else CompressionType.ZLIB
        }

    def compress(self, data: bytes, compression_type: CompressionType) -> Tuple[bytes, float]:
        compress_func = self.compression_map.get(compression_type, self._no_compress)
        compressed = compress_func(data)
        compression_ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
        return compressed, compression_ratio
        
    def smart_compress(self, data: bytes, data_type: str = 'regular_message', 
                      prioritize_speed: bool = True) -> Tuple[bytes, CompressionType, float]:
        """Intelligently select compression algorithm based on data type and priorities"""
        if len(data) < 100:
            # No need to compress small data
            return data, CompressionType.NONE, 1.0
            
        # Get recommended algorithm for this data type
        recommended_algo = self.data_type_mapping.get(data_type, CompressionType.ZLIB)
        
        # If prioritize_speed is True, choose the fastest available algorithm
        if prioritize_speed and lz4_available:
            best_algo = CompressionType.LZ4
        else:
            # Choose based on data type and algorithm properties
            best_algo = recommended_algo
        
        # Compress using the chosen algorithm
        compressed, ratio = self.compress(data, best_algo)
        
        # If compression doesn't help, return uncompressed
        if ratio > 0.95:
            return data, CompressionType.NONE, 1.0
            
        return compressed, best_algo, ratio

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
        
    def _lz4_compress(self, data: bytes) -> bytes:
        if lz4_available:
            return lz4.frame.compress(data)
        return data
        
    def _lz4_decompress(self, data: bytes) -> bytes:
        if lz4_available:
            return lz4.frame.decompress(data)
        return data
        
    def _brotli_compress(self, data: bytes) -> bytes:
        if brotli_available:
            return brotli.compress(data, quality=4)
        return data
        
    def _brotli_decompress(self, data: bytes) -> bytes:
        if brotli_available:
            return brotli.decompress(data)
        return data

    @staticmethod
    def is_zlib_compressed(data: bytes) -> bool:
        """Detect ZLIB magic bytes: 0x78 + {0x01, 0x5e, 0x9c, 0xda}"""
        return len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5e, 0x9c, 0xda)
        
    @staticmethod
    def is_lz4_compressed(data: bytes) -> bool:
        """Detect LZ4 magic bytes: 0x04 0x22 0x4D 0x18"""
        return len(data) >= 4 and data[:4] == b'\x04\x22\x4D\x18'
        
    @staticmethod
    def is_brotli_compressed(data: bytes) -> bool:
        """Detect Brotli magic bytes: 0x0B 0x79"""
        return len(data) >= 2 and data[:2] == b'\x0B\x79'

    def auto_decompress(self, data: bytes) -> bytes:
        """Auto-detect and decompress if data is compressed"""
        if len(data) < 2:
            return data
            
        if self.is_brotli_compressed(data) and brotli_available:
            try:
                return self._brotli_decompress(data)
            except Exception as e:
                logging.warning(f"Brotli auto-decompression failed, using raw data: {e}")
        elif self.is_lz4_compressed(data) and lz4_available:
            try:
                return self._lz4_decompress(data)
            except Exception as e:
                logging.warning(f"LZ4 auto-decompression failed, using raw data: {e}")
        elif self.is_zlib_compressed(data):
            try:
                return self._zlib_decompress(data)
            except Exception as e:
                logging.warning(f"ZLIB auto-decompression failed, using raw data: {e}")
        
        return data


class ConnectionPool:
    """Manages network connections efficiently"""

    def __init__(self, max_connections: int = 100, max_pipeline_connections: int = 20):
        self.max_connections = max_connections
        self.max_pipeline_connections = max_pipeline_connections
        self.active_connections: Dict[str, socket.socket] = {}
        self.connection_lock = threading.Lock()
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        # per-connection recv lock: prevents send_message ACK-reader and
        # start_receiving_from listener from racing on the same socket's recv
        self._recv_locks: Dict[str, threading.Lock] = {}
        # per-connection health check timestamp cache (5s)
        self._last_health_check: Dict[str, float] = {}
        
        # Pipeline-specific connection management
        self._pipeline_connections: Dict[str, socket.socket] = {}  # pipeline_id -> socket
        self._preheated_connections: Dict[str, List[socket.socket]] = {}  # endpoint_key -> list of preheated sockets
        self._pipeline_connection_usage: Dict[str, Dict[str, Any]] = {}  # pipeline_id -> usage stats
        
        # TCP pipeline optimization settings
        self._pipeline_flush_interval = 0.01  # 10ms flush interval for pipeline messages
        self._tcp_nodelay = True  # Disable Nagle for low latency
        self._socket_buffer_size = 524288  # 512KB buffer size for better throughput

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

    def _create_connection(self, endpoint: NetworkEndpoint, is_pipeline: bool = False) -> Optional[socket.socket]:
        """Create new connection with optimized socket parameters"""
        try:
            if endpoint.protocol == TransportProtocol.TCP:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(10.0)
                
                # TCP pipeline optimization settings
                nodelay = self._tcp_nodelay if is_pipeline else True  # Always disable Nagle for pipeline connections
                buffer_size = self._socket_buffer_size if is_pipeline else 262144
                
                # Socket 参数调优
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, nodelay)  # 禁用 Nagle 算法
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)    # 启用保活
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size) # send buffer
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size) # recv buffer
                
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
            
    def get_pipeline_connection(self, pipeline_id: str, endpoint: NetworkEndpoint) -> Optional[Tuple[socket.socket, threading.Lock]]:
        """Get or create a dedicated connection for a pipeline"""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"
        pipeline_key = f"{pipeline_id}:{connection_key}"
        
        with self.connection_lock:
            # Check if pipeline connection already exists
            if pipeline_key in self._pipeline_connections:
                conn = self._pipeline_connections[pipeline_key]
                if self._is_connection_alive_cached(conn, pipeline_key):
                    # Update usage stats
                    self._update_pipeline_usage(pipeline_key)
                    return conn, self._recv_locks.get(pipeline_key, threading.Lock())
                else:
                    # Remove dead connection
                    try:
                        conn.close()
                    except:
                        pass
                    del self._pipeline_connections[pipeline_key]
                    self._recv_locks.pop(pipeline_key, None)
                    self._last_health_check.pop(pipeline_key, None)
                    self._pipeline_connection_usage.pop(pipeline_key, None)
            
            # Check preheated connections first
            if connection_key in self._preheated_connections and self._preheated_connections[connection_key]:
                conn = self._preheated_connections[connection_key].pop()
                logging.info(f"Using preheated connection for pipeline {pipeline_id}")
            else:
                # Create new pipeline connection
                if len(self._pipeline_connections) >= self.max_pipeline_connections:
                    self._cleanup_pipeline_connections()
                
                conn = self._create_connection(endpoint, is_pipeline=True)
                if not conn:
                    return None
            
            # Store pipeline connection
            self._pipeline_connections[pipeline_key] = conn
            self._recv_locks[pipeline_key] = threading.Lock()
            self._last_health_check[pipeline_key] = time.time()
            
            # Initialize pipeline usage stats
            self._pipeline_connection_usage[pipeline_key] = {
                'created': time.time(),
                'last_used': time.time(),
                'message_count': 0,
                'bytes_sent': 0,
                'bytes_received': 0
            }
            
            return conn, self._recv_locks[pipeline_key]
            
    def preheat_connections(self, endpoint: NetworkEndpoint, count: int = 5):
        """Preheat connections for an endpoint to reduce connection establishment latency"""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"
        
        with self.connection_lock:
            if connection_key not in self._preheated_connections:
                self._preheated_connections[connection_key] = []
            
            # Create preheated connections
            for _ in range(count):
                conn = self._create_connection(endpoint, is_pipeline=True)
                if conn:
                    self._preheated_connections[connection_key].append(conn)
                    logging.info(f"Preheated connection for {connection_key}")
    
    def _update_pipeline_usage(self, pipeline_key: str):
        """Update pipeline connection usage statistics"""
        if pipeline_key in self._pipeline_connection_usage:
            self._pipeline_connection_usage[pipeline_key]['last_used'] = time.time()
            self._pipeline_connection_usage[pipeline_key]['message_count'] += 1
    
    def _cleanup_pipeline_connections(self):
        """Clean up idle pipeline connections"""
        current_time = time.time()
        to_remove = [
            key for key, stats in self._pipeline_connection_usage.items()
            if current_time - stats.get('last_used', 0) > 300  # 5 minutes idle
        ]
        
        for key in to_remove:
            try:
                if key in self._pipeline_connections:
                    self._pipeline_connections[key].close()
            except Exception as e:
                logging.debug(f"Error closing pipeline connection {key}: {e}")
            self._pipeline_connections.pop(key, None)
            self._recv_locks.pop(key, None)
            self._last_health_check.pop(key, None)
            self._pipeline_connection_usage.pop(key, None)

    def invalidate_connection(self, endpoint: NetworkEndpoint, pipeline_id: Optional[str] = None):
        """Forcibly remove a connection from pool (e.g., after error)"""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"
        with self.connection_lock:
            if pipeline_id:
                # Invalidate pipeline connection
                pipeline_key = f"{pipeline_id}:{connection_key}"
                if pipeline_key in self._pipeline_connections:
                    try:
                        self._pipeline_connections[pipeline_key].close()
                    except:
                        pass
                    del self._pipeline_connections[pipeline_key]
                    self._pipeline_connection_usage.pop(pipeline_key, None)
                    self._recv_locks.pop(pipeline_key, None)
                    self._last_health_check.pop(pipeline_key, None)
            else:
                # Invalidate regular connection
                if connection_key in self.active_connections:
                    try:
                        self.active_connections[connection_key].close()
                    except:
                        pass
                    del self.active_connections[connection_key]
                    self.connection_stats.pop(connection_key, None)
                    self._recv_locks.pop(connection_key, None)
                    self._last_health_check.pop(connection_key, None)

    def release_connection(self, endpoint: NetworkEndpoint, pipeline_id: Optional[str] = None):
        """Release connection back to pool (update last_used)"""
        connection_key = f"{endpoint.host}:{endpoint.port}:{endpoint.protocol.value}"
        with self.connection_lock:
            if pipeline_id:
                # Update pipeline connection usage
                pipeline_key = f"{pipeline_id}:{connection_key}"
                if pipeline_key in self._pipeline_connection_usage:
                    self._pipeline_connection_usage[pipeline_key]['last_used'] = time.time()
            else:
                # Update regular connection stats
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
                if key in self.active_connections:
                    self.active_connections[key].close()
            except Exception as e:
                logging.debug(f"Error closing connection {key}: {e}")
            self.active_connections.pop(key, None)
            self.connection_stats.pop(key, None)
            self._recv_locks.pop(key, None)
            self._last_health_check.pop(key, None)

    def close(self):
        """Explicitly close all connections and clean up resources"""
        with self.connection_lock:
            for key, conn in list(self.active_connections.items()):
                try:
                    conn.close()
                except Exception as e:
                    logging.debug(f"Error closing connection {key}: {e}")
            self.active_connections.clear()
            self.connection_stats.clear()
            self._recv_locks.clear()
            self._last_health_check.clear()
            self._pipeline_connections.clear()
            self._preheated_connections.clear()

    def __del__(self):
        """Clean up resources when the pool is garbage collected"""
        self.close()


class MessageRouter:
    """Routes messages between different platforms"""

    def __init__(self, hardware_capabilities: HardwareCapabilities):
        self.hardware_capabilities = hardware_capabilities
        self.message_handlers: Dict[str, List[Callable]] = {}  
        self.routing_table: Dict[str, NetworkEndpoint] = {}  
        self.message_queue = queue.Queue()  
        self.metrics: deque[MessageMetrics] = deque(maxlen=10000)  # 环形缓冲防止内存泄漏
        self.compression_manager = CompressionManager()
        self.serialization_manager = SerializationManager()
        self.connection_pool = ConnectionPool()
        
        # Thread pool configuration based on hardware capabilities
        self._configure_thread_pools()
        
        self.pipeline_comm_manager = PipelineCommunicationManager(self)

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
            
    def _configure_thread_pools(self):
        """Configure thread pools based on hardware capabilities"""
        platform = self.hardware_capabilities.platform
        # Use psutil if available, otherwise default to 4 cores
        cpu_count = psutil.cpu_count() if psutil_available else 4
        cpu_count = cpu_count or 4  # Fallback to 4 if psutil returns None
        
        # Determine thread pool sizes based on platform and CPU count
        if platform == HardwarePlatform.JETSON_NANO:
            # Resource-constrained platform
            handler_workers = max(2, cpu_count // 2)
            pipeline_workers = max(1, cpu_count // 4)
            serialization_workers = max(1, cpu_count // 4)
            compression_workers = max(1, cpu_count // 4)
        elif platform == HardwarePlatform.JETSON_ORIN:
            # More capable embedded platform
            handler_workers = max(4, cpu_count)
            pipeline_workers = max(2, cpu_count // 2)
            serialization_workers = max(2, cpu_count // 2)
            compression_workers = max(2, cpu_count // 2)
        elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_WINDOWS]:
            # High-performance platform
            handler_workers = max(8, cpu_count * 2)
            pipeline_workers = max(4, cpu_count)
            serialization_workers = max(4, cpu_count)
            compression_workers = max(4, cpu_count)
        else:
            # Default configuration
            handler_workers = max(4, cpu_count)
            pipeline_workers = max(2, cpu_count // 2)
            serialization_workers = max(2, cpu_count // 2)
            compression_workers = max(2, cpu_count // 2)
        
        # Create thread pools for different types of tasks
        # Note: thread_creation_flags is Linux-specific, so we don't use it for cross-platform compatibility
        self._handler_executor = ThreadPoolExecutor(
            max_workers=handler_workers, 
            thread_name_prefix="msg_handler"
        )
        
        self._pipeline_executor = ThreadPoolExecutor(
            max_workers=pipeline_workers, 
            thread_name_prefix="pipeline"
        )
        
        self._serialization_executor = ThreadPoolExecutor(
            max_workers=serialization_workers, 
            thread_name_prefix="serializer"
        )
        
        self._compression_executor = ThreadPoolExecutor(
            max_workers=compression_workers, 
            thread_name_prefix="compressor"
        )
        
        logging.info(f"Configured thread pools: handlers={handler_workers}, pipeline={pipeline_workers}, ")
        logging.info(f"serialization={serialization_workers}, compression={compression_workers}")
        
        # Task queue for background processing
        self._task_queue = queue.Queue(maxsize=10000)
        self._task_processing = True
        self._task_thread = threading.Thread(
            target=self._process_task_queue,
            name="task_processor",
            daemon=True
        )
        self._task_thread.start()
        
    def _process_task_queue(self):
        """Process background tasks from the task queue"""
        while self._task_processing:
            try:
                task = self._task_queue.get(timeout=0.1)
                if callable(task):
                    task()
                self._task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing task queue: {e}")
                self._task_queue.task_done()

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

    def close(self):
        """Explicitly close all resources"""
        # 关闭所有客户端连接
        with self._client_conn_lock:
            for key, conn in list(self.client_connections.items()):
                try:
                    conn.close()
                except Exception as e:
                    logging.debug(f"Error closing client connection {key}: {e}")
            self.client_connections.clear()

        # 停止监听线程
        for key in list(self._listener_running.keys()):
            self._listener_running[key] = False
        for key, thread in list(self._listener_threads.items()):
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._listener_threads.clear()
        self._listener_running.clear()

        # 关闭所有线程池
        executors = ['_handler_executor', '_pipeline_executor', '_serialization_executor', '_compression_executor']
        for executor_name in executors:
            if hasattr(self, executor_name):
                executor = getattr(self, executor_name)
                executor.shutdown(wait=True)

        # 停止任务队列处理
        if hasattr(self, '_task_processing'):
            self._task_processing = False
            if hasattr(self, '_task_thread') and self._task_thread.is_alive():
                self._task_thread.join(timeout=1.0)

        # 关闭连接池
        if hasattr(self, 'connection_pool'):
            self.connection_pool.close()

        # 清理队列
        if hasattr(self, 'message_queue'):
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except queue.Empty:
                    break

    def __del__(self):
        """Clean up resources when the router is garbage collected"""
        self.close()

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
                    requires_ack=False,  # Changed from True to avoid blocking
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
            # Start performance tracking
            start_total_time = time.time()
            
            # Serialize using appropriate format
            message_dict = message.to_dict()
            
            # Use endpoint's serialization format if specified, otherwise use message's format
            serialization_format = target_endpoint.serialization_format
            
            # Track serialization time
            start_serialization_time = time.time()
            message_data = self.serialization_manager.serialize(message_dict, serialization_format)
            serialization_time = time.time() - start_serialization_time

            # Smart compression based on message type
            data_type = 'regular_message'
            pipeline_id = None
            stage_id = None
            sequence_id = None
            sequence_index = 0
            
            if message.message_type == 'pipeline_data':
                # Determine pipeline data type from payload
                if isinstance(message.payload, dict):
                    if 'data_type' in message.payload:
                        data_type = f'pipeline_data_{message.payload["data_type"]}'
                    if 'pipeline_id' in message.payload:
                        pipeline_id = message.payload['pipeline_id']
                    if 'stage_id' in message.payload:
                        stage_id = message.payload['stage_id']
                    if 'sequence_id' in message.payload:
                        sequence_id = message.payload['sequence_id']
                    if 'sequence_index' in message.payload:
                        sequence_index = message.payload['sequence_index']
                else:
                    data_type = 'pipeline_data_activation'
            elif message.message_type == 'communication_bundle':
                data_type = 'batch_message'
            
            # Track compression time
            start_compression_time = time.time()
            message_data, compression_type, compression_ratio = self.compression_manager.smart_compress(
                message_data, data_type=data_type, prioritize_speed=True
            )
            compression_time = time.time() - start_compression_time

            # Track resource usage
            if psutil_available:
                cpu_usage = psutil.cpu_percent(interval=0.01)
                memory_usage = psutil.virtual_memory().percent
            else:
                cpu_usage = 0.0
                memory_usage = 0.0

            # Send data
            start_transmission_time = time.time()
            success = self._transmit_data(message_data, target_endpoint, message)
            transmission_time = time.time() - start_transmission_time

            # Record metrics
            metrics = MessageMetrics(
                message_id=message.message_id,
                source_id=message.source_id,
                target_id=message.target_id,
                timestamp=message.timestamp,
                size_bytes=len(message_data),
                compression_ratio=compression_ratio,
                transmission_time=transmission_time,
                success=success,
                protocol=target_endpoint.protocol,
                retry_count=0,
                
                # Pipeline-specific metrics
                pipeline_id=pipeline_id,
                stage_id=stage_id,
                sequence_id=sequence_id,
                sequence_index=sequence_index,
                data_type=data_type,
                
                # Performance metrics
                serialization_time=serialization_time,
                compression_time=compression_time,
                deserialization_time=0.0,
                decompression_time=0.0,
                queue_time=0.0,
                processing_time=time.time() - start_total_time,
                
                # Resource usage
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                
                # Compression details
                compression_algorithm=compression_type.value,
                serialization_format=serialization_format.value,
                
                # Connection details
                connection_type="regular",
                connection_reused=False
            )
            
            self.metrics.append(metrics)
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
        
        # Get current timeout setting
        original_timeout = conn.gettimeout()
        try:
            # Set a timeout for the entire recv operation
            if original_timeout is None:
                conn.settimeout(30.0)  # Default 30s timeout if none set
                
            while offset < n:
                try:
                    received = conn.recv_into(view[offset:], n - offset)
                    if received == 0:
                        return None
                    offset += received
                except socket.timeout:
                    logging.warning(f"Timeout waiting to receive {n} bytes")
                    return None
        finally:
            # Restore original timeout
            conn.settimeout(original_timeout)
            
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

                # 解析消息使用合适的序列化格式
                try:
                    message_dict = self.serialization_manager.deserialize(decompressed)
                    
                    # Handle batch messages
                    if isinstance(message_dict, dict) and 'batch_id' in message_dict:
                        try:
                            batch_message = BatchMessage.from_dict(message_dict)
                            logging.info(f"Received batch message {batch_message.batch_id} with {batch_message.batch_size} messages")
                            
                            # Process each message in the batch
                            for msg in batch_message.messages:
                                # --- listen_register：客户端专用接收 socket 的身份注册 ---
                                if msg.message_type == 'listen_register':
                                    source_node_id = msg.source_id
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
                                source_node_id = msg.source_id
                                with self._client_conn_lock:
                                    if source_node_id and source_node_id not in self.client_connections:
                                        self.client_connections[source_node_id] = client_socket
                                        logging.info(
                                            f"Registered client connection for {source_node_id} from {addr}")

                                # 发送 ACK
                                if msg.requires_ack:
                                    ack_payload = json.dumps(
                                        {'message_id': msg.message_id}).encode('utf-8')
                                    ack_len = len(ack_payload).to_bytes(4, byteorder='big')
                                    try:
                                        client_socket.sendall(ack_len + ack_payload)  # 合并为单次 sendall
                                    except Exception as e:
                                        logging.warning(f"Failed to send ACK to {source_node_id}: {e}")

                                # 分发消息
                                try:
                                    self._handle_message(msg)
                                except Exception as e:
                                    logging.error(f"Error handling message from {addr}: {e}")
                            
                            # If this was a listen_register batch, break the loop
                            if is_push_socket:
                                break
                            
                            continue
                        except Exception as e:
                            logging.error(f"Failed to process batch message: {e}")
                            continue
                    
                    # Handle single messages
                    try:
                        message = CrossPlatformMessage.from_dict(message_dict)
                    except ValueError as ve:
                        logging.warning(
                            f"_handle_tcp_connection({addr}): discarding malformed packet: {ve}"
                        )
                        continue
                except Exception as e:
                    logging.warning(
                        f"_handle_tcp_connection({addr}): failed to deserialize message: {e}"
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
                            
                            # 解析消息使用合适的序列化格式
                            try:
                                message_dict = self.serialization_manager.deserialize(decompressed)
                            except Exception as e:
                                logging.warning(
                                    f"start_receiving_from({node_id}): failed to deserialize message: {e}"
                                )
                                continue

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
        # Handle pipeline data messages first for sequence processing
        if message.message_type == "pipeline_data":
            # Use pipeline-specific executor for better performance
            self._pipeline_executor.submit(self.pipeline_comm_manager.handle_pipeline_message, message)
        
        # Still distribute to registered handlers for any additional processing
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
            
    def send_batch(self, messages: List[CrossPlatformMessage],
                   target_endpoint: NetworkEndpoint, 
                   source_id: Optional[str] = None) -> bool:
        """Send multiple messages in a single batch for efficient transmission"""
        if not messages:
            return True
            
        try:
            # Create batch message
            batch_id = f"batch_{int(time.time() * 1000)}_{id(messages)}"
            batch_source_id = source_id or messages[0].source_id
            batch_target_id = messages[0].target_id
            
            # Use the first message's compression type if not specified in endpoint
            compression_type = target_endpoint.compression
            
            # Create batch message
            batch_message = BatchMessage(
                batch_id=batch_id,
                source_id=batch_source_id,
                target_id=batch_target_id,
                messages=messages,
                compression_type=compression_type,
                serialization_format=target_endpoint.serialization_format
            )
            
            # Serialize and send batch
            batch_dict = batch_message.to_dict()
            batch_data = self.serialization_manager.serialize(batch_dict, target_endpoint.serialization_format)
            
            # Compress if needed
            if compression_type != CompressionType.NONE:
                batch_data, compression_ratio = self.compression_manager.compress(batch_data, compression_type)
            
            # Send batch
            result = self.connection_pool.get_connection(target_endpoint)
            if result is None:
                return False
            conn, recv_lock = result
            
            try:
                msg_len = len(batch_data).to_bytes(4, byteorder='big')
                conn.sendall(msg_len + batch_data)
                
                # Send ACK if requested
                if any(msg.requires_ack for msg in messages):
                    with recv_lock:
                        ack_len_data = self._recv_exact(conn, 4)
                        if not ack_len_data:
                            return False
                        ack_length = int.from_bytes(ack_len_data, byteorder='big')
                        ack_data = self._recv_exact(conn, ack_length)
                        if not ack_data:
                            return False
                    return True
                return True
            except Exception as e:
                logging.error(f"Failed to send batch: {e}")
                self.connection_pool.invalidate_connection(target_endpoint)
                return False
            finally:
                self.connection_pool.release_connection(target_endpoint)
                
        except Exception as e:
            logging.error(f"Failed to send batch message: {e}")
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
        
    def get_pipeline_performance_stats(self, pipeline_id: Optional[str] = None, time_window: float = 300.0) -> Dict[str, Any]:
        """Get detailed performance statistics for pipeline communication"""
        current_time = time.time()
        
        # Filter metrics based on pipeline_id and time window
        if pipeline_id:
            pipeline_metrics = [m for m in self.metrics 
                              if current_time - m.timestamp < time_window 
                              and m.pipeline_id == pipeline_id]
        else:
            pipeline_metrics = [m for m in self.metrics 
                              if current_time - m.timestamp < time_window 
                              and m.pipeline_id is not None]
        
        if not pipeline_metrics:
            return {}
            
        # Calculate overall statistics
        total_messages = len(pipeline_metrics)
        success_rate = sum(1 for m in pipeline_metrics if m.success) / total_messages
        avg_size_bytes = sum(m.size_bytes for m in pipeline_metrics) / total_messages
        avg_transmission_time_ms = sum(m.transmission_time for m in pipeline_metrics) / total_messages * 1000
        avg_serialization_time_ms = sum(m.serialization_time for m in pipeline_metrics) / total_messages * 1000
        avg_compression_time_ms = sum(m.compression_time for m in pipeline_metrics) / total_messages * 1000
        avg_processing_time_ms = sum(m.processing_time for m in pipeline_metrics) / total_messages * 1000
        avg_compression_ratio = sum(m.compression_ratio for m in pipeline_metrics) / total_messages
        avg_cpu_usage = sum(m.cpu_usage for m in pipeline_metrics) / total_messages
        avg_memory_usage = sum(m.memory_usage for m in pipeline_metrics) / total_messages
        
        # Calculate per-stage statistics
        stage_stats = {}
        for metric in pipeline_metrics:
            if not metric.stage_id:
                continue
                
            if metric.stage_id not in stage_stats:
                stage_stats[metric.stage_id] = {
                    "messages": [],
                    "total_time_ms": 0
                }
            
            stage_stats[metric.stage_id]["messages"].append(metric)
            stage_stats[metric.stage_id]["total_time_ms"] += metric.processing_time * 1000
        
        # Calculate per-stage averages
        for stage_id, stats in stage_stats.items():
            messages = stats["messages"]
            stage_stats[stage_id] = {
                "total_messages": len(messages),
                "success_rate": sum(1 for m in messages if m.success) / len(messages),
                "avg_processing_time_ms": stats["total_time_ms"] / len(messages),
                "avg_transmission_time_ms": sum(m.transmission_time for m in messages) / len(messages) * 1000,
                "avg_size_bytes": sum(m.size_bytes for m in messages) / len(messages),
                "avg_compression_ratio": sum(m.compression_ratio for m in messages) / len(messages)
            }
        
        # Calculate per-data-type statistics
        data_type_stats = {}
        for metric in pipeline_metrics:
            if not metric.data_type:
                continue
                
            if metric.data_type not in data_type_stats:
                data_type_stats[metric.data_type] = {
                    "messages": [],
                    "total_time_ms": 0
                }
            
            data_type_stats[metric.data_type]["messages"].append(metric)
            data_type_stats[metric.data_type]["total_time_ms"] += metric.processing_time * 1000
        
        # Calculate per-data-type averages
        for data_type, stats in data_type_stats.items():
            messages = stats["messages"]
            data_type_stats[data_type] = {
                "total_messages": len(messages),
                "success_rate": sum(1 for m in messages if m.success) / len(messages),
                "avg_processing_time_ms": stats["total_time_ms"] / len(messages),
                "avg_transmission_time_ms": sum(m.transmission_time for m in messages) / len(messages) * 1000,
                "avg_size_bytes": sum(m.size_bytes for m in messages) / len(messages),
                "avg_compression_ratio": sum(m.compression_ratio for m in messages) / len(messages)
            }
        
        # Calculate compression algorithm performance
        compression_stats = {}
        for metric in pipeline_metrics:
            if not metric.compression_algorithm:
                continue
                
            if metric.compression_algorithm not in compression_stats:
                compression_stats[metric.compression_algorithm] = {
                    "messages": [],
                    "total_compression_time_ms": 0
                }
            
            compression_stats[metric.compression_algorithm]["messages"].append(metric)
            compression_stats[metric.compression_algorithm]["total_compression_time_ms"] += metric.compression_time * 1000
        
        # Calculate per-compression-algorithm averages
        for algorithm, stats in compression_stats.items():
            messages = stats["messages"]
            compression_stats[algorithm] = {
                "total_messages": len(messages),
                "avg_compression_ratio": sum(m.compression_ratio for m in messages) / len(messages),
                "avg_compression_time_ms": stats["total_compression_time_ms"] / len(messages),
                "avg_processing_time_ms": sum(m.processing_time for m in messages) / len(messages) * 1000
            }
        
        return {
            "pipeline_id": pipeline_id,
            "time_window_seconds": time_window,
            "total_messages": total_messages,
            "success_rate": success_rate,
            "avg_size_bytes": avg_size_bytes,
            "avg_compression_ratio": avg_compression_ratio,
            "avg_transmission_time_ms": avg_transmission_time_ms,
            "avg_serialization_time_ms": avg_serialization_time_ms,
            "avg_compression_time_ms": avg_compression_time_ms,
            "avg_processing_time_ms": avg_processing_time_ms,
            "avg_cpu_usage_percent": avg_cpu_usage,
            "avg_memory_usage_percent": avg_memory_usage,
            "messages_per_second": total_messages / time_window,
            "bytes_per_second": sum(m.size_bytes for m in pipeline_metrics if m.success) / time_window,
            "stage_statistics": stage_stats,
            "data_type_statistics": data_type_stats,
            "compression_statistics": compression_stats
        }
        
    def print_pipeline_performance_report(self, pipeline_id: Optional[str] = None, time_window: float = 300.0):
        """Print a detailed performance report for pipeline communication"""
        stats = self.get_pipeline_performance_stats(pipeline_id, time_window)
        
        if not stats:
            print("No pipeline performance data available")
            return
            
        print(f"=== Pipeline Performance Report ===")
        print(f"Pipeline ID: {stats['pipeline_id'] or 'All Pipelines'}")
        print(f"Time Window: {stats['time_window_seconds']} seconds")
        print(f"Total Messages: {stats['total_messages']:.0f}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Messages per Second: {stats['messages_per_second']:.2f}")
        print(f"Bytes per Second: {stats['bytes_per_second'] / (1024 * 1024):.2f} MB/s")
        print(f"")
        print(f"=== Performance Metrics ===")
        print(f"Avg Transmission Time: {stats['avg_transmission_time_ms']:.2f} ms")
        print(f"Avg Serialization Time: {stats['avg_serialization_time_ms']:.2f} ms")
        print(f"Avg Compression Time: {stats['avg_compression_time_ms']:.2f} ms")
        print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.2f} ms")
        print(f"Avg Message Size: {stats['avg_size_bytes'] / 1024:.2f} KB")
        print(f"Avg Compression Ratio: {stats['avg_compression_ratio']:.2f}x")
        print(f"Avg CPU Usage: {stats['avg_cpu_usage_percent']:.1f}%")
        print(f"Avg Memory Usage: {stats['avg_memory_usage_percent']:.1f}%")
        
        if stats['stage_statistics']:
            print(f"\n=== Stage Statistics ===")
            for stage_id, stage_stat in sorted(stats['stage_statistics'].items()):
                print(f"Stage {stage_id}:")
                print(f"  Messages: {stage_stat['total_messages']}")
                print(f"  Success Rate: {stage_stat['success_rate']:.2%}")
                print(f"  Avg Processing Time: {stage_stat['avg_processing_time_ms']:.2f} ms")
                print(f"  Avg Message Size: {stage_stat['avg_size_bytes'] / 1024:.2f} KB")
                
        if stats['data_type_statistics']:
            print(f"\n=== Data Type Statistics ===")
            for data_type, data_stat in stats['data_type_statistics'].items():
                print(f"{data_type}:")
                print(f"  Messages: {data_stat['total_messages']}")
                print(f"  Avg Processing Time: {data_stat['avg_processing_time_ms']:.2f} ms")
                print(f"  Avg Message Size: {data_stat['avg_size_bytes'] / 1024:.2f} KB")
                print(f"  Avg Compression Ratio: {data_stat['avg_compression_ratio']:.2f}x")
                
        if stats['compression_statistics']:
            print(f"\n=== Compression Algorithm Performance ===")
            for algorithm, comp_stat in stats['compression_statistics'].items():
                print(f"{algorithm}:")
                print(f"  Messages: {comp_stat['total_messages']}")
                print(f"  Avg Compression Ratio: {comp_stat['avg_compression_ratio']:.2f}x")
                print(f"  Avg Compression Time: {comp_stat['avg_compression_time_ms']:.2f} ms")
                print(f"  Avg Processing Time: {comp_stat['avg_processing_time_ms']:.2f} ms")
                
        print(f"\n====================================")
        
    def export_pipeline_performance_data(self, pipeline_id: Optional[str] = None, time_window: float = 300.0) -> Dict[str, Any]:
        """Export pipeline performance data for external analysis"""
        stats = self.get_pipeline_performance_stats(pipeline_id, time_window)
        
        if not stats:
            return {}
            
        # Convert to export format
        export_data = {
            "metadata": {
                "export_time": time.time(),
                "pipeline_id": stats["pipeline_id"],
                "time_window_seconds": stats["time_window_seconds"]
            },
            "summary": {
                "total_messages": stats["total_messages"],
                "success_rate": stats["success_rate"],
                "messages_per_second": stats["messages_per_second"],
                "bytes_per_second": stats["bytes_per_second"],
                "avg_transmission_time_ms": stats["avg_transmission_time_ms"],
                "avg_processing_time_ms": stats["avg_processing_time_ms"],
                "avg_compression_ratio": stats["avg_compression_ratio"]
            },
            "detailed_metrics": [
                {
                    "message_id": m.message_id,
                    "timestamp": m.timestamp,
                    "pipeline_id": m.pipeline_id,
                    "stage_id": m.stage_id,
                    "sequence_id": m.sequence_id,
                    "sequence_index": m.sequence_index,
                    "data_type": m.data_type,
                    "size_bytes": m.size_bytes,
                    "compression_ratio": m.compression_ratio,
                    "transmission_time_ms": m.transmission_time * 1000,
                    "serialization_time_ms": m.serialization_time * 1000,
                    "compression_time_ms": m.compression_time * 1000,
                    "processing_time_ms": m.processing_time * 1000,
                    "compression_algorithm": m.compression_algorithm,
                    "serialization_format": m.serialization_format,
                    "connection_type": m.connection_type,
                    "success": m.success,
                    "cpu_usage_percent": m.cpu_usage,
                    "memory_usage_percent": m.memory_usage
                }
                for m in self.metrics 
                if current_time - m.timestamp < time_window 
                and (m.pipeline_id == pipeline_id or pipeline_id is None)
                and m.pipeline_id is not None
            ]
        }
        
        return export_data


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

            # Get CPU load if psutil is available, otherwise default to 0.5
            cpu_load = psutil.cpu_percent() / 100.0 if psutil_available else 0.5
            
            if result == 0:
                latency = (time.time() - start_time) * 1000
                quality = max(0.0, min(1.0, 100.0 / max(latency, 1)))
                bandwidth = 1000.0 if endpoint.host.startswith('192.168.') else 100.0
                return {
                    'quality': quality,
                    'latency': latency,
                    'bandwidth': bandwidth,
                    'cpu_load': cpu_load
                }
            else:
                return {'quality': 0.0, 'latency': 9999.0,
                        'bandwidth': 0.1, 'cpu_load': cpu_load}
        except Exception:
            return {'quality': 0.1, 'latency': 1000.0,
                    'bandwidth': 10.0, 'cpu_load': cpu_load}

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
