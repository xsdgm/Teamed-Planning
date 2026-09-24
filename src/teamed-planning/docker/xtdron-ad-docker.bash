#!/bin/bash

# =================================================================
# Image: natsuago/xtdrone_ard:260209
# Features: Auto-GPU, Mandatory GUI, Auto-Cleanup, Color Support
# =================================================================

CONTAINER_NAME=${1:-xtdrone-ad}
IMAGE_NAME="xtdrone_ad:v4"

# 1. Mandatory GUI Check
if [ -z "$DISPLAY" ]; then
    echo "Error: No DISPLAY detected! GUI is required for Gazebo/XTDrone."
    exit 1
else
    xhost +local:docker > /dev/null 2>&1
fi

# 2. Automatic GPU Detection
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "Check: NVIDIA GPU detected. Enabling GPU support."
    GPU_FLAG="--gpus all"
else
    echo "Check: No NVIDIA GPU detected. Using CPU mode."
fi

# 3. Handle Container Lifecycle (Start/Fix/Create)
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Info: Container '$CONTAINER_NAME' exists."
    
    # Try to start and enter
    if ! docker start "$CONTAINER_NAME" 2>/dev/null; then
        echo "Warning: Failed to start. Container configuration might be incompatible."
        read -p "Recreate container? (Existing data in system folders will be lost) [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            docker rm -f "$CONTAINER_NAME"
        else
            exit 1
        fi
    else
        # Enter existing container with color support
        docker exec -it -e TERM=xterm-256color "$CONTAINER_NAME" /bin/bash -l
        exit 0
    fi
fi

# 4. Create New Container with Safe Color Injection
echo "Action: Creating new container: $CONTAINER_NAME"

docker run $GPU_FLAG -it \
  --privileged \
  --net=host \
  --name "$CONTAINER_NAME" \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$SSH_AUTH_SOCK":/ssh-agent \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY="$XAUTHORITY" \
  -e SSH_AUTH_SOCK=/ssh-agent \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e TERM=xterm-256color \
  -e force_color_prompt=yes \
  -e GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o User=git -o StrictHostKeyChecking=accept-new' \
  "$IMAGE_NAME" \
  /bin/bash -lbin/bash
