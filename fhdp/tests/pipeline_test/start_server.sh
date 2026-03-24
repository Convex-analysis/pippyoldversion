#!/bin/bash
# Start server in background with logging

LOG_FILE="server.log"
PID_FILE="server.pid"

# Kill existing server if running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Killing existing server (PID: $OLD_PID)"
        kill $OLD_PID
        sleep 1
    fi
fi

# Start server in background
echo "Starting server..."
cd "$(dirname "$0")"
nohup ./run_pipeline_test.sh server > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo $SERVER_PID > "$PID_FILE"
echo "Server started with PID: $SERVER_PID"
echo "Log file: $LOG_FILE"

# Wait for server to start
sleep 3

# Check if server is running
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✓ Server is running"
    echo "Check logs with: tail -f $LOG_FILE"
else
    echo "✗ Server failed to start. Check log file: $LOG_FILE"
    cat "$LOG_FILE"
    rm "$PID_FILE"
    exit 1
fi
