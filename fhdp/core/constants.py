"""
FHDP System Constants and Configuration Parameters
"""

# System Performance Requirements
TEMPLATE_LOOKUP_LATENCY_THRESHOLD = 0.005  # 5ms maximum lookup latency
PIPELINE_RECOMPOSITION_TIME = 1.5  # 1.5s maximum recomposition time
TRAINING_EPOCHS_SHORT = (1, 2)  # Short-horizon training epochs range

# Communication Configuration
DEFAULT_COMMUNICATION_PROTOCOL = "dsrc"
COMMUNICATION_BUNDLE_SIZE = 64 * 1024  # 64KB bundles
MAX_NEIGHBOR_DISTANCE = 300.0  # meters for V2V communication
MIN_SIGNAL_STRENGTH = -85.0  # dBm minimum signal strength

# Mobility Prediction Parameters
DTMC_PREDICTION_HORIZON = 10.0  # seconds
DTMC_TRANSITION_MEMORY = 5  # number of previous states to consider
MOBILITY_UPDATE_INTERVAL = 1.0  # seconds
POSITION_TOLERANCE = 5.0  # meters position tolerance

# Template and Pipeline Configuration
TEMPLATE_CACHE_SIZE = 1000
MAX_PIPELINE_LENGTH = 5
MIN_PIPELINE_PARTICIPANTS = 2
PIPELINE_TIMEOUT = 30.0  # seconds before pipeline dissolution
TEMPLATE_GENERATION_INTERVAL = 60.0  # seconds

# Resource Classification Thresholds
HIGH_RESOURCE_CPU = 0.8  # 80% CPU availability
HIGH_RESOURCE_MEMORY = 0.8  # 80% memory availability
HIGH_RESOURCE_BATTERY = 0.7  # 70% battery

MEDIUM_RESOURCE_CPU = 0.5  # 50% CPU availability
MEDIUM_RESOURCE_MEMORY = 0.5  # 50% memory availability
MEDIUM_RESOURCE_BATTERY = 0.4  # 40% battery

# Fairness Mechanism Parameters
FAIRNESS_WINDOW_SIZE = 100  # number of recent participations to consider
FAIRNESS_DECAY_FACTOR = 0.9  # decay factor for historical participation
MIN_PARTICIPATION_INTERVAL = 5.0  # minimum seconds between participations
MAX_PRIORITY_WEIGHT = 3.0  # maximum priority weight multiplier

# Aggregation Configuration
ASYNC_AGGREGATION_INTERVAL = 2.0  # seconds
MIN_AGGREGATION_PARTICIPANTS = 3
AGGREGATION_TIMEOUT = 10.0  # seconds
WEIGHT_DECAY_FACTOR = 0.95  # for model update weights

# Error Propagation Configuration
ERROR_ACCUMULATION_THRESHOLD = 0.1  # threshold for lazy propagation
MAX_ERROR_PROPAGATION_DELAY = 5.0  # seconds
ERROR_COMPRESSION_RATIO = 0.7  # compression for error signals

# V2V Communication Protocols
PROTOCOL_BANDWIDTH = {
    "dsrc": 27.0,  # Mbps
    "cv2x": 100.0,  # Mbps  
    "wifi_direct": 250.0  # Mbps
}

PROTOCOL_LATENCY = {
    "dsrc": 0.01,  # 10ms
    "cv2x": 0.005,  # 5ms
    "wifi_direct": 0.002  # 2ms
}

PROTOCOL_RANGE = {
    "dsrc": 300.0,  # meters
    "cv2x": 500.0,  # meters
    "wifi_direct": 200.0  # meters
}

# Training Parameters
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOCAL_EPOCHS = 1
MAX_LOCAL_DATA_SIZE = 10000
MIN_LOCAL_DATA_SIZE = 100

# Memory and Performance Constraints
MAX_VEHICLE_MEMORY_USAGE = 0.8  # 80% of available memory
MAX_TEMPLATE_MEMORY = 100 * 1024 * 1024  # 100MB for template storage
MAX_PIPELINE_STATE_MEMORY = 50 * 1024 * 1024  # 50MB for pipeline states

# System Monitoring and Logging
MONITORING_INTERVAL = 5.0  # seconds
LOG_LEVEL = "INFO"
METRICS_COLLECTION_INTERVAL = 10.0  # seconds

# Security and Reliability
MAX_CONNECTION_ATTEMPTS = 3
CONNECTION_TIMEOUT = 3.0  # seconds
HEARTBEAT_INTERVAL = 2.0  # seconds
MISSING_HEARTBEAT_THRESHOLD = 3  # consecutive missing heartbeats