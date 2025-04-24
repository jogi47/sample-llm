# Pyth API

A simple FastAPI project with GET and POST endpoints using strict typing, featuring a local LLM model optimized for macOS.

## Setup

1. Install dependencies:
```bash
pip install -e .
```

## Type Checking

This project uses strict typing with mypy. To run type checks:

```bash
mypy main.py
```

## Running the API

You can run the server directly:

```bash
python main.py
```

For macOS users (especially with Apple Silicon), use the optimized script:

```bash
./run_macos.sh
```

The script automatically detects your available RAM and selects the best model.

The API will be available at http://localhost:8000

## Interactive API Documentation & Playground

This project comes with built-in interactive API documentation:

- **Swagger UI**: Available at http://localhost:8000/docs
  - Interactive playground to test all endpoints
  - Execute API calls directly from your browser
  - Test the LLM generation with different parameters
  - View model schemas and response formats

- **ReDoc**: Available at http://localhost:8000/redoc
  - Clean, responsive documentation
  - Better for reading and understanding the API structure

When you start the server and navigate to http://localhost:8000, you'll be redirected to a documentation page that links to both documentation systems.

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

You can also modify the `config.py` file to add more models or adjust settings.

## API Endpoints

### GET /welcome
Returns a welcome message.

### GET /items
Returns all items in the collection.

### POST /items
Add a new item to the collection.

Example request body:
```json
{
  "name": "Item name",
  "description": "Item description",
  "value": 100
}
```

### POST /llm/generate
Generate a response from the local LLM model.

Example request body:
```json
{
  "prompt": "What is the capital of France?",
  "max_length": 512,
  "temperature": 0.7,
  "top_p": 0.95
}
```

### GET /llm/info
Get information about the current LLM model and available models.

## Testing the LLM

You can test the LLM endpoint with the included test script:

```bash
python test_llm.py
```

Or use the interactive Swagger UI at http://localhost:8000/docs to test all endpoints directly in your browser.
