#!/bin/bash
# Stop Edge TPU inference server

CONTAINER_NAME="edgetpu-server"

echo "Stopping Edge TPU server..."
docker stop "$CONTAINER_NAME" 2>/dev/null && echo "Server stopped" || echo "Server not running"
docker rm "$CONTAINER_NAME" 2>/dev/null || true

