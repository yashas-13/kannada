# Architecture Overview

## High-Level Flow (Local-Only)
```
Browser (UI) --> Flask app --> Local LLM (Ollama/LM Studio)
                 |        \
                 |         -> SQLite (plans, users, settings)
                 -> System Status API (local-only + TTS availability)
```

## Sequence: Pricing + Local-Only Banner
1. Browser loads `/` and requests `/api/plans` + `/api/subscription` to render pricing cards.
2. Browser requests `/api/system/status` to show local-only mode + TTS availability.
3. User upgrades plan via `/api/subscribe`, which updates the local SQLite DB.

## Notes
- The system enforces local-only backends via `LOCAL_ONLY_MODE=1` (default).
- TTS/ASR models must be cached locally; the app will not download models in local-only mode.
