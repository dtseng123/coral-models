# Coral Edge TPU Inference Server

High-performance ML inference server for **dual Coral Edge TPUs** on Raspberry Pi 5. Provides socket-based API for real-time object detection and pose estimation. 

## Performance

| Task | Inference Time | FPS |
|------|---------------|-----|
| Object Detection (SSD MobileNet v2) | ~21ms | 47 |
| Pose Estimation (MoveNet Lightning) | ~12ms | 82 |

## Hardware Setup

### Components
- **Raspberry Pi 5** (8GB)
- **Dual M.2 HAT** (s2pi or similar)
- **Dual Coral Edge TPU M.2 Adapter** (for dual-TPU module)
- **Dual Coral Edge TPU M.2 A+E** 
- **NVMe SSD** (for root filesystem)
- **MicroSD Card** (for boot - required workaround)

### Boot Configuration

The dual M.2 HAT with NVMe + Coral TPU won't boot directly from NVMe. The workaround is to boot from SD card with NVMe as rootfs:

1. **SD Card Boot Partition** (`/dev/mmcblk0p1` → `/boot/firmware`)
2. **NVMe Root Filesystem** (`/dev/nvme0n1p2` → `/`)

#### SD Card cmdline.txt
```
console=serial0,115200 console=tty1 root=UUID=<your-nvme-uuid> rootfstype=ext4 fsck.repair=yes pcie_aspm=off rootwait
```

#### /boot/firmware/config.txt additions
```ini
# Coral Edge TPU - Use 4KB page kernel (REQUIRED)
kernel=kernel8.img

# Enable PCIe
dtparam=pciex1
dtparam=pciex1_gen=2

# Coral Edge TPU PCIe support
dtoverlay=pcie-32bit-dma-pi5
dtoverlay=pciex1-compat-pi5,no-mip
```

> **Important**: The `kernel=kernel8.img` line is critical - it forces the 4KB page kernel. The default Pi 5 16KB kernel is incompatible with libedgetpu.

### Driver Installation

```bash
# Add Coral repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install drivers
sudo apt update
sudo apt install -y gasket-dkms libedgetpu1-std

# Verify TPUs detected
ls /dev/apex*
# Should show: /dev/apex_0  /dev/apex_1
```

## Server Setup

### Prerequisites
- Docker installed
- Coral Edge TPU drivers loaded (`/dev/apex_0`, `/dev/apex_1`)

### Download Models

```bash
cd /home/dash/coral-models
mkdir -p models

# SSD MobileNet v2 (object detection)
curl -L -o models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite

# MoveNet Lightning (pose estimation)
curl -L -o models/movenet_lightning_edgetpu.tflite \
  https://github.com/google-coral/test_data/raw/master/movenet_single_pose_lightning_ptq_edgetpu.tflite

# EfficientDet Lite0 (optional - better detection)
curl -L -o models/efficientdet_lite0_edgetpu.tflite \
  https://github.com/google-coral/test_data/raw/master/efficientdet_lite0_edgetpu.tflite
```

### Build & Run

```bash
# Build Docker image
./build.sh

# Run server
./run.sh

# Or install as systemd service (auto-start on boot)
sudo cp edgetpu-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now edgetpu-server
```

### Verify

```bash
# Check service status
sudo systemctl status edgetpu-server

# View logs
sudo docker logs -f edgetpu-server

# Test inference
python3 edgetpu_client.py
```

## API Usage

The server exposes two socket-based services:
- **Detection**: `localhost:5590`
- **Pose Estimation**: `localhost:5591`

### Python Client

```python
from edgetpu_client import EdgeTPUClient

client = EdgeTPUClient()

# Object detection
with open('image.jpg', 'rb') as f:
    result = client.detect(f.read())
    
for det in result['detections']:
    print(f"{det['class_name']}: {det['confidence']:.0%} at {det['bbox']}")

# Pose estimation
with open('image.jpg', 'rb') as f:
    keypoints = client.get_keypoints(f.read())
    
if 'left_wrist' in keypoints:
    print(f"Left hand at ({keypoints['left_wrist']['x']}, {keypoints['left_wrist']['y']})")
```

### From NumPy/OpenCV

```python
import cv2
from edgetpu_client import EdgeTPUClient

client = EdgeTPUClient()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = client.detect(frame)  # Accepts numpy arrays directly
    
    for det in result['detections']:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

## Related Projects

This inference server was built to support:

### [optidex](https://github.com/dtseng123/optidex)
Portable voice controlled AI platform using computer vision for object detection and tracking. [Based on Whisplay AI Chatbot](https://github.com/PiSugar/whisplay-ai-chatbot)

### [vr-passthrough](https://github.com/dtseng123/vr-passthrough)
VR passthrough/ mixed reality system with real-time object detection and pose estimation for controller-free interaction.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Host (Pi 5)                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Docker Container                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ Detection Server│  │  Pose Server    │       │   │
│  │  │   (port 5590)   │  │  (port 5591)    │       │   │
│  │  │                 │  │                 │       │   │
│  │  │  SSD MobileNet  │  │    MoveNet      │       │   │
│  │  │   + pycoral     │  │   + pycoral     │       │   │
│  │  └────────┬────────┘  └────────┬────────┘       │   │
│  │           │                    │                 │   │
│  │           ▼                    ▼                 │   │
│  │     /dev/apex_0          /dev/apex_1            │   │
│  └─────────────────────────────────────────────────┘   │
│                    │                │                   │
│         ┌─────────┴────────────────┴─────────┐         │
│         │         Dual M.2 HAT               │         │
│         │   ┌──────────┐  ┌──────────┐       │         │
│         │   │ Coral #1 │  │ Coral #2 │       │         │
│         │   │ (TPU 0)  │  │ (TPU 1)  │       │         │
│         │   └──────────┘  └──────────┘       │         │
│         └────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### TPUs not detected (`/dev/apex_*` missing)
```bash
# Check PCIe devices
lspci | grep -i coral

# Check kernel modules
lsmod | grep apex

# Rebuild gasket driver
sudo apt install --reinstall gasket-dkms
sudo modprobe apex
```

### "Could not map pages" error
The kernel page size is wrong. Ensure `kernel=kernel8.img` is in `/boot/firmware/config.txt` and reboot.

```bash
# Verify 4KB pages
getconf PAGE_SIZE
# Should output: 4096
```

### Inference slow or failing in Docker
Make sure container has device access:
```bash
sudo docker run --privileged --device /dev/apex_0 --device /dev/apex_1 ...
```

## License

MIT

