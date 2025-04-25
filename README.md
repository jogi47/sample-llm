# Pyth API

A simple FastAPI project with a chat interface and API endpoints, featuring a local LLM model optimized for macOS.

## Features

- **Chat Interface**: Multi-threaded chat UI similar to popular AI chat applications
- **Dark/Light Theme**: Toggle between dark and light modes
- **Local LLM Integration**: Run AI models directly on your machine
- **Model Switching**: Change between different models on-the-fly
- **API Endpoints**: Access LLM functionality programmatically

## Setup with UV

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. If you don't have uv installed:

```bash
# Install uv (using pip)
pip install uv
```

### Installing Dependencies

```bash
# Install dependencies using uv
uv pip install -e .
```

## Type Checking

This project uses strict typing with mypy. To run type checks:

```bash
uv run mypy main.py
```

## Running the Application

### Development Mode

```bash
# Run in development mode with auto-reload
python main.py
```

For macOS users (especially with Apple Silicon), use the optimized script:

```bash
./run_macos.sh
```

The script automatically detects your available RAM and selects the best model.

The application will be available at:
- Chat Interface: http://localhost:8000/
- API Documentation: http://localhost:8000/docs

### Production Mode

For production deployment, use the provided production script:

```bash
./run_production.sh
```

## Using the Chat Interface

1. **Access the Chat Interface**: Open your browser and go to http://localhost:8000/
2. **Create New Conversations**: Click "New Chat" to start a new thread
3. **Switch Between Threads**: Click on any thread in the sidebar to switch contexts
4. **Change Models**: Use the dropdown menu in the top-right to switch between models
5. **Toggle Dark/Light Mode**: Click the moon/sun icon to change the theme

## API Endpoints

### Chat-related Endpoints

- `GET /`: Chat interface
- `POST /api/chat`: Generate a chat response
- `POST /api/set-model`: Change the active model

### Other API Endpoints

- `GET /welcome`: Returns a welcome message
- `GET /items`: Returns all items in the collection
- `POST /items`: Add a new item to the collection
- `POST /llm/generate`: Generate a response from the LLM model
- `GET /llm/info`: Get information about available models

## Mac Compatibility

This project has been optimized to work on macOS with Apple Silicon (M1/M2/M3). It uses:

- MPS (Metal Performance Shaders) when available for GPU acceleration
- Models that are compatible with 8GB RAM on macOS
- Memory optimizations for efficient inference

## LLM Model Configuration

You can select which LLM model to use by setting the `LLM_MODEL` environment variable:

```bash
# Use the tiny model (default, suitable for systems with limited RAM)
LLM_MODEL=tiny python main.py

# Use the small model (better capabilities but requires more RAM)
LLM_MODEL=small python main.py

# Use the medium model (best capabilities on 8GB RAM)
LLM_MODEL=medium python main.py
```

Available models:
- `tiny`: TinyLlama-1.1B-Chat-v1.0 (works on 4-8GB RAM)
- `small`: bigscience/bloom-560m (works on 4-8GB RAM)
- `medium`: microsoft/phi-2 (works on 8GB+ RAM)

## Model Fine-tuning

See [TRAINING.md](TRAINING.md) for information on fine-tuning models with custom datasets.

## Testing the API

You can test the API endpoints with the interactive Swagger UI at http://localhost:8000/docs
