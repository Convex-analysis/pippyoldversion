#!/bin/bash
#
# FHDP Pipeline Training Test Launcher (Refactored)
#
# This script helps launch pipeline training test on different machines.
# Now uses FHDP's built-in cross_platform_comm for reliable network communication.
#
# Usage:
#   ./run_pipeline_test_refactored.sh [server|agx|nano] [server-ip]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     FHDP Pipeline Training Test Launcher                  ║"
    echo "║     Using FHDP's cross_platform_comm (Refactored)        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo "Usage: $0 [MODE] [SERVER_IP]"
    echo ""
    echo "Modes:"
    echo "  server  - Run on 4090 Linux server (coordinator)"
    echo "  agx     - Run on Jetson AGX Orin (high resource vehicle)"
    echo "  nano    - Run on Jetson Orin Nano (medium resource vehicle)"
    echo ""
    echo "Arguments:"
    echo "  SERVER_IP  - IP address of Mac proxy (required for agx and nano modes)"
    echo ""
    echo "Examples:"
    echo "  # On 4090 server:"
    echo "  $0 server"
    echo ""
    echo "  # On Jetson AGX Orin (connect via Mac proxy):"
    echo "  $0 agx 219.216.65.34"
    echo ""
    echo "  # On Jetson Orin Nano (connect via Mac proxy):"
    echo "  $0 nano 219.216.65.34"
    echo ""
    echo "Prerequisites:"
    echo "  1. On Mac: Establish SSH port forwarding:"
    echo "     ssh -L 0.0.0.0:5001:localhost:5001 xta@219.216.64.173 -N -f"
    echo ""
    echo "  2. On Linux 4090 server: Start the server first:"
    echo "     $0 server"
    echo ""
    echo "  3. On Jetson devices: Connect to Mac proxy IP, not Linux server directly"
    echo ""
    echo "Environment Variables:"
    echo "  FHDP_PORT      - Server port (default: 5001)"
    echo "  FHDP_CONFIG    - Path to config file"
    echo "  PYTHON         - Python interpreter (default: python3)"
    echo ""
    echo "Network Architecture:"
    echo "  Jetson -> Mac (219.216.65.34:5001) -> Linux 4090 (219.216.64.173:5001)"
    echo "  Mac acts as SSH port forwarder for cross-subnet communication"
}

# Check Python installation
check_python() {
    PYTHON_CMD=${PYTHON:-python3}

    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Using Python: $($PYTHON_CMD --version)${NC}"
}

# Check PyTorch installation
check_pytorch() {
    PYTHON_CMD=${PYTHON:-python3}

    if ! $PYTHON_CMD -c "import torch" &> /dev/null; then
        echo -e "${YELLOW}Warning: PyTorch not found. Installing...${NC}"
        echo "Please install PyTorch before running the test."
        echo "For Jetson devices, refer to: https://developer.nvidia.com/embedded/downloads"
        exit 1
    fi

    # Check CUDA availability
    CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())")
    echo -e "${GREEN}✓ PyTorch installed, CUDA available: $CUDA_AVAILABLE${NC}"
}

# Run server mode
run_server() {
    local port=${FHDP_PORT:-5001}
    local config=${FHDP_CONFIG:-""}

    echo -e "${BLUE}"
    echo "=================================="
    echo "Starting Server Mode"
    echo "=================================="
    echo -e "${NC}"
    echo "Host: 0.0.0.0"
    echo "Port: $port"
    echo "Config: ${config:-(default)}"
    echo ""
    echo -e "${GREEN}Using FHDP's built-in cross_platform_comm for reliable communication${NC}"
    echo "Features: Length-prefix protocol, zlib compression, connection pooling"
    echo ""

    local cmd="$PYTHON_CMD test_pipeline_training_refactored.py --mode server --host 0.0.0.0 --port $port"

    if [ -n "$config" ]; then
        cmd="$cmd --config $config"
    fi

    echo "Executing: $cmd"
    echo ""

    exec $cmd
}

# Run vehicle mode
run_vehicle() {
    local mode=$1
    local server_ip=$2
    local port=${FHDP_PORT:-5001}
    local config=${FHDP_CONFIG:-""}

    # Set vehicle parameters based on mode
    if [ "$mode" = "agx" ]; then
        local vehicle_id="agx_orin_001"
        local resource_level="high"
    elif [ "$mode" = "nano" ]; then
        local vehicle_id="orin_nano_001"
        local resource_level="medium"
    else
        echo -e "${RED}Error: Invalid mode '$mode'${NC}"
        exit 1
    fi

    echo -e "${BLUE}"
    echo "=================================="
    echo "Starting Vehicle Mode"
    echo "=================================="
    echo -e "${NC}"
    echo "Vehicle ID: $vehicle_id"
    echo "Server: $server_ip:$port"
    echo "Resource Level: $resource_level"
    echo "Config: ${config:-(default)}"
    echo ""
    echo -e "${GREEN}Using FHDP's built-in cross_platform_comm for reliable communication${NC}"
    echo "Features: Length-prefix protocol, zlib compression, connection pooling"
    echo ""

    local cmd="$PYTHON_CMD test_pipeline_training_refactored.py --mode vehicle --vehicle-id $vehicle_id --server-host $server_ip --server-port $port --resource-level $resource_level"

    if [ -n "$config" ]; then
        cmd="$cmd --config $config"
    fi

    echo "Executing: $cmd"
    echo ""

    exec $cmd
}

# Main execution
main() {
    print_banner

    # Check arguments
    if [ $# -lt 1 ]; then
        print_usage
        exit 1
    fi

    MODE=$1
    SERVER_IP=${2:-""}

    # Check dependencies
    check_python
    check_pytorch

    # Execute based on mode
    case $MODE in
        server)
            if [ -n "$SERVER_IP" ]; then
                echo -e "${YELLOW}Warning: SERVER_IP ignored in server mode${NC}"
            fi
            run_server
            ;;
        agx|nano)
            if [ -z "$SERVER_IP" ]; then
                echo -e "${RED}Error: SERVER_IP is required for $MODE mode${NC}"
                echo ""
                echo "Please provide the Mac proxy IP address (219.216.65.34)"
                print_usage
                exit 1
            fi
            run_vehicle $MODE $SERVER_IP
            ;;
        help|--help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Invalid mode '$MODE'${NC}"
            print_usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"
