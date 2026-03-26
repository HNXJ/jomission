# mlxEngine (Local)

A lightweight, robust FastAPI wrapper for `mlx-lm`, providing an OpenAI-compatible chat completions endpoint optimized for Apple Silicon (M-series).

## Features
- **OpenAI Compatibility**: Supports `/v1/chat/completions` (no-streaming yet).
- **VRAM Aware**: Automatic VRAM clearing and management via `mlx.core.metal`.
- **Configurable**: Simple `config.json` for paths, ports, and API keys.
- **Auto-load**: Remembers the last loaded model for instant availability on restart.

## Quick Start

### 1. Requirements
Ensure you have the following installed:
```bash
pip install fastapi uvicorn mlx-lm mlx psutil pydantic
```

### 2. Configuration
Edit `config.json` to match your environment:
- `model_root`: Relative or absolute path to your MLX models directory.
- `port`: Port to run the server on (default: `4474`).
- `api_key`: Authorization token for requests.
- `memory_ceiling_gb`: Safe limit for VRAM usage.

### 3. Run
```bash
python mlx-lm-engine-control.py
```

## API Usage

### Status Check
```bash
curl http://localhost:4474/status
```

### Chat Completion (OpenAI Compatible)
```bash
curl -X POST http://localhost:4474/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Your-Model-Name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

## Maintenance
Logs are written to `engine.log` and status updates to `engine-status.md`.
