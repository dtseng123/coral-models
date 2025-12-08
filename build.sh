#!/bin/bash
# Build Edge TPU inference server Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Building Edge TPU Inference Server"
echo "========================================"
echo ""

# Create models directory if needed
mkdir -p models

# Copy models to models directory
echo "Setting up models..."
for model in ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
             efficientdet_lite0_edgetpu.tflite \
             movenet_lightning_edgetpu.tflite; do
    if [ -f "$model" ] && [ ! -f "models/$model" ]; then
        cp "$model" "models/"
        echo "  Copied $model"
    fi
done

# Also copy labels
if [ -f "coco_labels.txt" ] && [ ! -f "models/coco_labels.txt" ]; then
    cp coco_labels.txt models/
fi

echo ""
echo "Building Docker image..."
docker build -t edgetpu-server .

echo ""
echo "========================================"
echo "Build complete!"
echo ""
echo "To run the server:"
echo "  ./run.sh"
echo ""
echo "To test the client:"
echo "  python3 edgetpu_client.py"
echo "========================================"

