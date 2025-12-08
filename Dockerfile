# Edge TPU Inference Service for Raspberry Pi 5
# Uses Debian Bullseye with Python 3.9 for pycoral compatibility

FROM debian:bullseye-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add Coral repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install Edge TPU runtime and PyCoral
RUN apt-get update && apt-get install -y --no-install-recommends \
    libedgetpu1-std \
    python3-pycoral \
    python3-tflite-runtime \
    python3-numpy \
    python3-pillow \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy inference server
COPY edgetpu_server.py /app/
COPY models/ /app/models/

# Expose ports for socket API
# 5590 = Detection service
# 5591 = Pose estimation service
EXPOSE 5590 5591

# Run the inference server
CMD ["python3", "/app/edgetpu_server.py"]

