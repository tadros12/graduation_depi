#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "============================================="
echo "   Launching SentinelVision Land Cover AI    "
echo "============================================="

# Move to deployment directory
cd "$SCRIPT_DIR/deployment"

# Free port 8000 if occupied by a stale process
if lsof -t -i:8000 >/dev/null 2>&1; then
    echo "Freeing port 8000 from stale process..."
    kill -9 $(lsof -t -i:8000) 2>/dev/null || true
    sleep 1
fi

# Start uvicorn server in the background on port 8000
/home/theodoros/graduation/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# Wait for the server to spin up
echo "Waiting for server to start..."
sleep 2

# Automatically open default web browser
if command -v xdg-open > /dev/null; then
    echo "Opening web dashboard in browser..."
    xdg-open http://127.0.0.1:8000
fi

echo ""
echo "✅ Server is running on http://127.0.0.1:8000"
echo "============================================="
echo "Press Ctrl+C in this terminal to stop the server."
echo "============================================="

# Handle graceful shutdown on Ctrl+C (SIGINT / SIGTERM)
cleanup() {
    echo ""
    echo "Stopping SentinelVision server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    echo "Goodbye!"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for the server process
wait $SERVER_PID
