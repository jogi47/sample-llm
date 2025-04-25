#!/bin/bash
set -e

echo "Running macOS-optimized LLM API server..."

# Set environment variables for better macOS performance
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Check available RAM and select appropriate model
TOTAL_RAM_GB=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
echo "Detected $TOTAL_RAM_GB GB of RAM"

if [ "$TOTAL_RAM_GB" -lt 8 ]; then
    echo "Less than 8GB RAM detected, using 'tiny' model"
    export LLM_MODEL=tiny
elif [ "$TOTAL_RAM_GB" -lt 16 ]; then
    echo "8-16GB RAM detected, using 'medium' model"
    export LLM_MODEL=tiny
else
    echo "16GB+ RAM detected, using 'medium' model"
    export LLM_MODEL=medium
fi

# Run type checking with uv
echo "Running type checking..."
uv pip mypy main.py

# Start the server in development mode
echo "Starting API server (development mode)..."
uv pip python main.py 