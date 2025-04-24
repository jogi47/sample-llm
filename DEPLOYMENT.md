# Deployment Guide

This document provides instructions for deploying the Pyth API to production environments.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- 8GB+ RAM recommended for the medium model

## Installation with uv

1. Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd pyth
```

2. Install dependencies with uv:

```bash
uv pip install -e .
```

3. For production environments, install extra requirements:

```bash
uv pip install "uvicorn[standard]"
```

## Production Deployment Options

### Option 1: Using the Production Script

The simplest deployment method is to use the included production script:

```bash
./run_production.sh
```

This script:
- Automatically detects available RAM and selects an appropriate model
- Sets MacOS-specific optimizations if needed
- Runs uvicorn with production settings
- Uses a single worker for LLM processing

### Option 2: Manual Configuration

If you need more control, you can configure production settings manually:

```bash
# Set environment variables
export LLM_MODEL=medium  # Choose model based on available RAM

# For MacOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run with production settings
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
```

### Option 3: System Service (Linux)

For Linux servers, create a systemd service:

1. Create a service file at `/etc/systemd/system/pyth.service`:

```ini
[Unit]
Description=Pyth API Service
After=network.target

[Service]
User=<your-user>
WorkingDirectory=/path/to/pyth
Environment="LLM_MODEL=medium"
ExecStart=/path/to/pyth/run_production.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

2. Enable and start the service:

```bash
sudo systemctl enable pyth
sudo systemctl start pyth
```

## Testing the Production Deployment

Verify the API is running correctly:

```bash
curl http://localhost:8000/welcome
```

Test the LLM endpoint:

```bash
curl -X POST http://localhost:8000/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of France?"}'
```

## Monitoring and Logs

For monitoring and logs in production:

1. View logs with systemd (if using Linux service):

```bash
sudo journalctl -u pyth -f
```

2. For manual process monitoring, consider using tools like:
   - `htop` or `top` for CPU/memory usage
   - `nvidia-smi` if using GPU

## Recommended Model by Environment

| Environment | RAM  | Recommended Model |
|-------------|------|------------------|
| Development | Any  | tiny             |
| Small VPS   | <8GB | tiny             |
| Standard Server | 8-16GB | medium     |
| High-memory Server | >16GB | medium   |

## Troubleshooting

If you encounter memory issues:
- Set `LLM_MODEL=tiny` for constrained environments
- Update `config.py` to enable `LOW_MEMORY_MODE = True` 