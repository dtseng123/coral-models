#!/bin/bash
# Run Edge TPU inference server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Container name
CONTAINER_NAME="edgetpu-server"

# Check if already running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo "Edge TPU server is already running"
    echo "To stop: docker stop $CONTAINER_NAME"
    echo "To view logs: docker logs -f $CONTAINER_NAME"
    exit 0
fi

# Remove old container if exists
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "========================================"
echo "Starting Edge TPU Inference Server"
echo "========================================"
echo ""

# Check TPU devices
echo "Checking TPU devices..."
if [ -e /dev/apex_0 ]; then
    echo "  ✓ /dev/apex_0 found"
else
    echo "  ✗ /dev/apex_0 not found!"
    exit 1
fi

if [ -e /dev/apex_1 ]; then
    echo "  ✓ /dev/apex_1 found"
else
    echo "  ⚠ /dev/apex_1 not found (will use single TPU)"
fi

echo ""
echo "Starting container..."

# Run container
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --device /dev/apex_0:/dev/apex_0 \
    --device /dev/apex_1:/dev/apex_1 \
    -p 5590:5590 \
    -p 5591:5591 \
    -v "$SCRIPT_DIR/models:/app/models:ro" \
    edgetpu-server

echo ""
echo "Waiting for server to start..."
sleep 3

# Check if running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo ""
    echo "========================================"
    echo "Edge TPU Server is running!"
    echo ""
    echo "Ports:"
    echo "  Detection: localhost:5590"
    echo "  Pose:      localhost:5591"
    echo ""
    echo "Commands:"
    echo "  View logs:  docker logs -f $CONTAINER_NAME"
    echo "  Stop:       docker stop $CONTAINER_NAME"
    echo "  Restart:    docker restart $CONTAINER_NAME"
    echo ""
    echo "Test with:"
    echo "  python3 edgetpu_client.py"
    echo "========================================"
    
    # Show initial logs
    echo ""
    echo "Server logs:"
    docker logs "$CONTAINER_NAME"
else
    echo "Failed to start server!"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

