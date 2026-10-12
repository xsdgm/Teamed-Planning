#!/usr/bin/env bash
set -euo pipefail

if [[ ! -c /dev/dxg || ! -S /tmp/.X11-unix/X0 ]]; then
    echo 'WSLg GPU device or X11 display is missing. Check Docker Desktop WSL 2 and WSLg mounts.' >&2
    exit 1
fi
if ! command -v glxinfo >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    if ! apt-get install -y --no-install-recommends mesa-utils >/tmp/mesa-utils-install.log 2>&1; then
        cat /tmp/mesa-utils-install.log >&2
        exit 1
    fi
fi
glxinfo -B | tee /tmp/wslg-renderer.log
if ! grep -q 'Accelerated: yes' /tmp/wslg-renderer.log ||
        ! grep -Eq 'OpenGL renderer string: D3D12 \(NVIDIA' /tmp/wslg-renderer.log; then
    echo 'NVIDIA hardware rendering is not active; refusing to start with software rendering.' >&2
    exit 1
fi
