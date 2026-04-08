#!/bin/bash

# EdgePipe Jetson Run Script
# This script simplifies the process of running EdgePipe on two Jetson devices

# Default configuration
SERVER_HOST="0.0.0.0"
SERVER_PORT="5000"
DEVICE0_ID="orin"
DEVICE1_ID="nano"
DEVICE0_HOST=""
DEVICE1_HOST=""
ROUNDS="1"
MICRO_BATCHES="4"
TEMPLATE_ID="vit_b16_2stage_v1"
DATA_DIR="./data"
DOWNLOAD="false"
AUTO_EXIT="true"

# Colors for output
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Usage function
usage() {
    echo -e "${GREEN}Usage:${NC} $0 [options] [command]"
    echo -e ""
    echo -e "${GREEN}Commands:${NC}"
    echo -e "  server          Start the EdgePipe server"
    echo -e "  device0         Start EdgePipe on device0 (Jetson Orin)"
    echo -e "  device1         Start EdgePipe on device1 (Jetson Nano)"
    echo -e "  all             Start server and both devices (requires ssh access)"
    echo -e "  status          Check status of EdgePipe system"
    echo -e "  cleanup         Clean up EdgePipe processes"
    echo -e ""
    echo -e "${GREEN}Options:${NC}"
    echo -e "  --server-host   Server host address (default: $SERVER_HOST)"
    echo -e "  --server-port   Server port (default: $SERVER_PORT)"
    echo -e "  --device0-id    Device0 ID (default: $DEVICE0_ID)"
    echo -e "  --device1-id    Device1 ID (default: $DEVICE1_ID)"
    echo -e "  --device0-host  Device0 host address (required for 'all' command)"
    echo -e "  --device1-host  Device1 host address (required for 'all' command)"
    echo -e "  --rounds        Number of training rounds (default: $ROUNDS)"
    echo -e "  --micro-batches Number of micro-batches (default: $MICRO_BATCHES)"
    echo -e "  --template-id   Model template ID (default: $TEMPLATE_ID)"
    echo -e "  --data-dir      Data directory (default: $DATA_DIR)"
    echo -e "  --download      Download dataset if not present (default: $DOWNLOAD)"
    echo -e "  --auto-exit     Automatically exit after training (default: $AUTO_EXIT)"
    echo -e "  -h, --help      Show this help message"
    echo -e ""
    echo -e "${GREEN}Examples:${NC}"
    echo -e "  # Start server"
    echo -e "  $0 server"
    echo -e ""
    echo -e "  # Start device0"
    echo -e "  $0 device0 --server-host 192.168.1.100"
    echo -e ""
    echo -e "  # Start all components (server + both devices)"
    echo -e "  $0 all --device0-host 192.168.1.101 --device1-host 192.168.1.102"
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            server|device0|device1|all|status|cleanup)
                COMMAND="$1"
                shift
                ;;
            --server-host)
                SERVER_HOST="$2"
                shift 2
                ;;
            --server-port)
                SERVER_PORT="$2"
                shift 2
                ;;
            --device0-id)
                DEVICE0_ID="$2"
                shift 2
                ;;
            --device1-id)
                DEVICE1_ID="$2"
                shift 2
                ;;
            --device0-host)
                DEVICE0_HOST="$2"
                shift 2
                ;;
            --device1-host)
                DEVICE1_HOST="$2"
                shift 2
                ;;
            --rounds)
                ROUNDS="$2"
                shift 2
                ;;
            --micro-batches)
                MICRO_BATCHES="$2"
                shift 2
                ;;
            --template-id)
                TEMPLATE_ID="$2"
                shift 2
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --download)
                DOWNLOAD="$2"
                shift 2
                ;;
            --auto-exit)
                AUTO_EXIT="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo -e "${RED}Error: Unknown argument $1${NC}"
                usage
                exit 1
                ;;
        esac
    done

    # Validate command
    if [[ -z "$COMMAND" ]]; then
        echo -e "${RED}Error: No command specified${NC}"
        usage
        exit 1
    fi

    # Validate required arguments for 'all' command
    if [[ "$COMMAND" == "all" ]]; then
        if [[ -z "$DEVICE0_HOST" || -z "$DEVICE1_HOST" ]]; then
            echo -e "${RED}Error: --device0-host and --device1-host are required for 'all' command${NC}"
            usage
            exit 1
        fi
    fi
}

# Start server
start_server() {
    echo -e "${GREEN}Starting EdgePipe server...${NC}"
    echo -e "Server host: ${SERVER_HOST}:${SERVER_PORT}"
    echo -e "Device IDs: ${DEVICE0_ID}, ${DEVICE1_ID}"
    echo -e "Rounds: ${ROUNDS}"
    echo -e "Micro-batches: ${MICRO_BATCHES}"
    echo -e "Template ID: ${TEMPLATE_ID}"
    echo -e ""

    python -m fhdp.EdgePipe.edgepipe_jetson \
        --mode server \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        --device0-id "${DEVICE0_ID}" \
        --device1-id "${DEVICE1_ID}" \
        --rounds "${ROUNDS}" \
        --micro-batches "${MICRO_BATCHES}" \
        --template-id "${TEMPLATE_ID}" \
        --auto-exit "${AUTO_EXIT}"
}

# Start device0
start_device0() {
    echo -e "${GREEN}Starting EdgePipe device0...${NC}"
    echo -e "Device ID: ${DEVICE0_ID}"
    echo -e "Server: ${SERVER_HOST}:${SERVER_PORT}"
    echo -e "Rounds: ${ROUNDS}"
    echo -e "Micro-batches: ${MICRO_BATCHES}"
    echo -e "Data directory: ${DATA_DIR}"
    echo -e "Download: ${DOWNLOAD}"
    echo -e ""

    python -m fhdp.EdgePipe.edgepipe_jetson \
        --mode device \
        --role device0 \
        --device-id "${DEVICE0_ID}" \
        --server-host "${SERVER_HOST}" \
        --server-port "${SERVER_PORT}" \
        --device0-id "${DEVICE0_ID}" \
        --device1-id "${DEVICE1_ID}" \
        --listen-host "0.0.0.0" \
        --listen-port "5001" \
        --rounds "${ROUNDS}" \
        --micro-batches "${MICRO_BATCHES}" \
        --data-dir "${DATA_DIR}" \
        --download "${DOWNLOAD}" \
        --auto-exit "${AUTO_EXIT}"
}

# Start device1
start_device1() {
    echo -e "${GREEN}Starting EdgePipe device1...${NC}"
    echo -e "Device ID: ${DEVICE1_ID}"
    echo -e "Server: ${SERVER_HOST}:${SERVER_PORT}"
    echo -e "Rounds: ${ROUNDS}"
    echo -e "Micro-batches: ${MICRO_BATCHES}"
    echo -e "Data directory: ${DATA_DIR}"
    echo -e "Download: ${DOWNLOAD}"
    echo -e ""

    python -m fhdp.EdgePipe.edgepipe_jetson \
        --mode device \
        --role device1 \
        --device-id "${DEVICE1_ID}" \
        --server-host "${SERVER_HOST}" \
        --server-port "${SERVER_PORT}" \
        --device0-id "${DEVICE0_ID}" \
        --device1-id "${DEVICE1_ID}" \
        --listen-host "0.0.0.0" \
        --listen-port "5002" \
        --rounds "${ROUNDS}" \
        --micro-batches "${MICRO_BATCHES}" \
        --data-dir "${DATA_DIR}" \
        --download "${DOWNLOAD}" \
        --auto-exit "${AUTO_EXIT}"
}

# Start all components
start_all() {
    echo -e "${GREEN}Starting EdgePipe system (server + both devices)...${NC}"
    echo -e "Server: ${SERVER_HOST}:${SERVER_PORT}"
    echo -e "Device0: ${DEVICE0_ID}@${DEVICE0_HOST}"
    echo -e "Device1: ${DEVICE1_ID}@${DEVICE1_HOST}"
    echo -e ""

    # Start server in background
    echo -e "${YELLOW}Starting server...${NC}"
    start_server &
    SERVER_PID=$!
    sleep 3

    # Start device0 via ssh
    echo -e "${YELLOW}Starting device0...${NC}"
    ssh "${DEVICE0_HOST}" "cd $(pwd) && bash run_edgepipe_jetson.sh device0 --server-host ${SERVER_HOST} --server-port ${SERVER_PORT} --device0-id ${DEVICE0_ID} --device1-id ${DEVICE1_ID} --rounds ${ROUNDS} --micro-batches ${MICRO_BATCHES} --data-dir ${DATA_DIR} --download ${DOWNLOAD} --auto-exit ${AUTO_EXIT}" &
    DEVICE0_PID=$!

    # Start device1 via ssh
    echo -e "${YELLOW}Starting device1...${NC}"
    ssh "${DEVICE1_HOST}" "cd $(pwd) && bash run_edgepipe_jetson.sh device1 --server-host ${SERVER_HOST} --server-port ${SERVER_PORT} --device0-id ${DEVICE0_ID} --device1-id ${DEVICE1_ID} --rounds ${ROUNDS} --micro-batches ${MICRO_BATCHES} --data-dir ${DATA_DIR} --download ${DOWNLOAD} --auto-exit ${AUTO_EXIT}" &
    DEVICE1_PID=$!

    # Wait for processes to finish
    wait $SERVER_PID $DEVICE0_PID $DEVICE1_PID
}

# Check status
check_status() {
    echo -e "${GREEN}Checking EdgePipe system status...${NC}"
    
    # Check if server is running
    SERVER_RUNNING=$(ps aux | grep "edgepipe_jetson.*--mode server" | grep -v grep | wc -l)
    if [[ $SERVER_RUNNING -gt 0 ]]; then
        echo -e "${GREEN}✓ Server is running${NC}"
    else
        echo -e "${RED}✗ Server is not running${NC}"
    fi
    
    # Check if device0 is running
    DEVICE0_RUNNING=$(ps aux | grep "edgepipe_jetson.*--role device0" | grep -v grep | wc -l)
    if [[ $DEVICE0_RUNNING -gt 0 ]]; then
        echo -e "${GREEN}✓ Device0 is running${NC}"
    else
        echo -e "${RED}✗ Device0 is not running${NC}"
    fi
    
    # Check if device1 is running
    DEVICE1_RUNNING=$(ps aux | grep "edgepipe_jetson.*--role device1" | grep -v grep | wc -l)
    if [[ $DEVICE1_RUNNING -gt 0 ]]; then
        echo -e "${GREEN}✓ Device1 is running${NC}"
    else
        echo -e "${RED}✗ Device1 is not running${NC}"
    fi
}

# Cleanup processes
cleanup() {
    echo -e "${GREEN}Cleaning up EdgePipe processes...${NC}"
    
    # Kill server process
    SERVER_PID=$(ps aux | grep "edgepipe_jetson.*--mode server" | grep -v grep | awk '{print $2}')
    if [[ ! -z "$SERVER_PID" ]]; then
        echo -e "${YELLOW}Killing server process: $SERVER_PID${NC}"
        kill -9 $SERVER_PID 2>/dev/null
    fi
    
    # Kill device0 process
    DEVICE0_PID=$(ps aux | grep "edgepipe_jetson.*--role device0" | grep -v grep | awk '{print $2}')
    if [[ ! -z "$DEVICE0_PID" ]]; then
        echo -e "${YELLOW}Killing device0 process: $DEVICE0_PID${NC}"
        kill -9 $DEVICE0_PID 2>/dev/null
    fi
    
    # Kill device1 process
    DEVICE1_PID=$(ps aux | grep "edgepipe_jetson.*--role device1" | grep -v grep | awk '{print $2}')
    if [[ ! -z "$DEVICE1_PID" ]]; then
        echo -e "${YELLOW}Killing device1 process: $DEVICE1_PID${NC}"
        kill -9 $DEVICE1_PID 2>/dev/null
    fi
    
    echo -e "${GREEN}Cleanup completed${NC}"
}

# Main function
main() {
    parse_args "$@"
    
    case "$COMMAND" in
        server)
            start_server
            ;;
        device0)
            start_device0
            ;;
        device1)
            start_device1
            ;;
        all)
            start_all
            ;;
        status)
            check_status
            ;;
        cleanup)
            cleanup
            ;;
        *)
            echo -e "${RED}Error: Unknown command $COMMAND${NC}"
            usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"
