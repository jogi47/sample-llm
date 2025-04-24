#!/bin/bash
set -e

echo "Running type checking..."
mypy main.py

echo "Starting API server..."
python main.py 