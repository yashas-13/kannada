# Kannada AI Assistant

**Status:** Active commercial prototype (local-only, self-serve pricing enabled)  
**Version:** 0.2.1

This is a Flask web application that acts as a frontend for a local Kannada LLM (`gemma-kannada-instruct`) running at `http://192.168.80.1:1234`.

## Features
- **Chat Mode**: General conversation with the AI.
- **Translation Mode**: Specialized prompts for English -> Kannada and Kannada -> English translation.
- **YouTube Workflow**: Download, transcribe, and translate English videos.
- **Commercial Plans**: Self-serve pricing tiers with plan-aware limits.
- **Local Only**: Does not rely on external APIs (except for Bootstrap CDN, which is cached by browsers).

## Latest Changes
- Enforced local-only backend usage and surfaced system status in the UI.
- Added system status API for local-only visibility and TTS availability.
- Updated pricing UI to show local-only banner messaging.
- Removed demo/test scripts to keep the repo production-focused.

## Prerequisites
1. Ensure your local LLM server is running at `http://192.168.80.1:1234`.
2. Python 3.8+ installed.

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`.

## Database / Migration Notes
This release extends the `plans` table with `summary` and `features_json`, and adds a `settings` table for system flags. If you are upgrading an existing deployment, start the app once to apply the automatic schema update.

## Configuration
To change the LLM endpoint or model name, edit `app.py`:

```python
LLM_API_URL = "http://192.168.80.1:1234/v1/chat/completions"
MODEL_NAME = "gemma-kannada-instruct"
```

Local-only enforcement is on by default. To allow non-local backends, set `LOCAL_ONLY_MODE=0` (not recommended for offline use).

## API Examples (cURL)
### List plans
```bash
curl http://localhost:5000/api/plans
```

### Get current subscription
```bash
curl http://localhost:5000/api/subscription
```

### Subscribe to a plan
```bash
curl -X POST http://localhost:5000/api/subscribe \
  -H "Content-Type: application/json" \
  -d '{"plan_name": "pro"}'
```

### System status (local-only + TTS availability)
```bash
curl http://localhost:5000/api/system/status
```

## Architecture
See `docs/ARCHITECTURE.md` for the local-only flow and pricing sequence overview.
