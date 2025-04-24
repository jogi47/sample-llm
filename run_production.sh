#!/bin/bash
set -e

echo "Starting Pyth API in production mode..."

# Set environment variables
export PYTHONPATH=$(pwd)

# Check available RAM and select appropriate model
TOTAL_RAM_GB=$(sysctl hw.memsize 2>/dev/null | awk '{print int($2/1024/1024/1024)}' || free -g | awk 'NR==2 {print $2}')
echo "Detected $TOTAL_RAM_GB GB of RAM"

if [ "$TOTAL_RAM_GB" -lt 8 ]; then
    echo "Less than 8GB RAM detected, using 'tiny' model"
    export LLM_MODEL=tiny
elif [ "$TOTAL_RAM_GB" -lt 16 ]; then
    echo "8-16GB RAM detected, using 'medium' model"
    export LLM_MODEL=medium
else
    echo "16GB+ RAM detected, using 'medium' model"
    export LLM_MODEL=medium
fi

# Set performance variables for MacOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "MacOS detected, setting MPS optimizations"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

# Ensure uvicorn[standard] is installed
echo "Ensuring uvicorn[standard] is installed..."
uv pip install "uvicorn[standard]" --quiet

# Start the server in production mode
echo "Starting production server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log 