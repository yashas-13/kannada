from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, g, abort
import requests
import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile
import io
import numpy as np
import time
import os
import shutil
import threading
import uuid
from pathlib import Path
import re
import json
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from urllib.parse import urlparse
from werkzeug.security import generate_password_hash, check_password_hash

from yt_dlp import YoutubeDL
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
except Exception:
    YouTubeTranscriptApi = None

app = Flask(__name__)

# Configuration
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto").strip().lower()
LOCAL_ONLY_MODE = os.environ.get("LOCAL_ONLY_MODE", "1").strip() not in ("0", "false", "False")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "auto")

ASR_MODEL_ID = os.environ.get("ASR_MODEL_ID", "distil-whisper/distil-small.en")

TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-kan")
LLM_CONNECT_TIMEOUT_SECONDS = 10
LLM_READ_TIMEOUT_SECONDS = 600
LLM_MAX_RETRIES = 2
LLM_RETRY_BACKOFF_SECONDS = 2

APP_DATA_DIR = Path(os.environ.get("KANNADA_TRANSLATOR_DATA_DIR", str(Path(__file__).parent / "data"))).resolve()
JOBS_DIR = (APP_DATA_DIR / "jobs").resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = (APP_DATA_DIR / "app.db").resolve()

JOBS_LOCK = threading.Lock()
JOBS = {}

# Lazy-loaded ASR
ASR_LOCK = threading.Lock()
ASR_PIPELINE = None
LMSTUDIO_MODEL_ID = None
LLM_ACTIVE_BACKEND = None

BILINGUAL_LOCKS = {}


def _get_or_create_secret_key():
    key_path = (APP_DATA_DIR / "secret_key.txt").resolve()
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    env_key = (os.environ.get("FLASK_SECRET_KEY") or "").strip()
    if env_key:
        key_path.write_text(env_key, encoding="utf-8")
        return env_key
    key = uuid.uuid4().hex + uuid.uuid4().hex
    key_path.write_text(key, encoding="utf-8")
    return key


app.secret_key = _get_or_create_secret_key()


def _get_db():
    db = getattr(g, "_db", None)
    if db is None:
        db = sqlite3.connect(str(DB_PATH))
        db.row_factory = sqlite3.Row
        g._db = db
    return db


@app.teardown_appcontext
def _close_db(exception):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()


def _init_db():
    db = sqlite3.connect(str(DB_PATH))
    try:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                monthly_char_limit INTEGER,
                monthly_youtube_minutes_limit INTEGER,
                price_cents INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                plan_id INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(id)
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_monthly (
                user_id INTEGER NOT NULL,
                year_month TEXT NOT NULL,
                chars_used INTEGER NOT NULL DEFAULT 0,
                youtube_minutes_used INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, year_month),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS youtube_jobs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                model_id TEXT,
                status TEXT NOT NULL,
                error TEXT,
                duration_seconds INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        db.commit()
    finally:
        db.close()


def _ensure_schema():
    db = sqlite3.connect(str(DB_PATH))
    try:
        db.row_factory = sqlite3.Row
        columns = {row["name"] for row in db.execute("PRAGMA table_info(plans)").fetchall()}
        if "summary" not in columns:
            # WHY: commercial plans need marketing copy; WHAT: COMM-001; HOW: drop column to rollback.
            db.execute("ALTER TABLE plans ADD COLUMN summary TEXT")
        if "features_json" not in columns:
            # WHY: store tier features for UI rendering; WHAT: COMM-001; HOW: remove column if reverting pricing cards.
            db.execute("ALTER TABLE plans ADD COLUMN features_json TEXT")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        db.commit()
    finally:
        db.close()


def _ensure_default_plans():
    db = sqlite3.connect(str(DB_PATH))
    try:
        db.row_factory = sqlite3.Row
        existing = {r["name"] for r in db.execute("SELECT name FROM plans").fetchall()}
        defaults = [
            (
                "free",
                20000,
                30,
                0,
                "Starter access for personal use",
                json.dumps(
                    [
                        "20k characters/month",
                        "30 YouTube minutes/month",
                        "Community support",
                        "Basic model access",
                    ]
                ),
            ),
            (
                "pro",
                200000,
                300,
                149900,
                "For teams shipping multilingual content",
                json.dumps(
                    [
                        "200k characters/month",
                        "300 YouTube minutes/month",
                        "Priority email support",
                        "Advanced model access",
                        "Usage analytics",
                    ]
                ),
            ),
            (
                "business",
                None,
                None,
                499900,
                "Unlimited usage for enterprises",
                json.dumps(
                    [
                        "Unlimited characters",
                        "Unlimited YouTube minutes",
                        "Dedicated success manager",
                        "Custom model routing",
                        "SLA-backed support",
                    ]
                ),
            ),
        ]
        for name, char_limit, yt_limit, price_cents, summary, features_json in defaults:
            if name in existing:
                db.execute(
                    """
                    UPDATE plans
                    SET summary = COALESCE(summary, ?),
                        features_json = COALESCE(features_json, ?)
                    WHERE name = ?
                    """,
                    (summary, features_json, name),
                )
                continue
            db.execute(
                """
                INSERT INTO plans (
                    name,
                    monthly_char_limit,
                    monthly_youtube_minutes_limit,
                    price_cents,
                    summary,
                    features_json,
                    is_active
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (name, char_limit, yt_limit, price_cents, summary, features_json),
            )
        db.commit()
    finally:
        db.close()


def _ensure_default_settings():
    db = sqlite3.connect(str(DB_PATH))
    try:
        local_only_value = "1" if LOCAL_ONLY_MODE else "0"
        db.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            ("local_only_mode", local_only_value),
        )
        # WHY: keep system flags in sync for the UI; WHAT: COMM-002; HOW: remove settings table usage to roll back.
        db.execute("UPDATE settings SET value = ? WHERE key = ?", (local_only_value, "local_only_mode"))
        db.commit()
    finally:
        db.close()


def _get_setting(key, default=None):
    db = _get_db()
    row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def _validate_local_backend_urls():
    if not LOCAL_ONLY_MODE:
        return

    def _is_local(url):
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "0.0.0.0"}

    if not _is_local(OLLAMA_BASE_URL):
        # WHY: enforce local-only usage with no paid/external APIs; WHAT: COMM-002; HOW: allow remote by disabling LOCAL_ONLY_MODE.
        raise RuntimeError("LOCAL_ONLY_MODE requires OLLAMA_BASE_URL to be localhost")
    if not _is_local(LMSTUDIO_BASE_URL):
        # WHY: enforce local-only usage with no paid/external APIs; WHAT: COMM-002; HOW: allow remote by disabling LOCAL_ONLY_MODE.
        raise RuntimeError("LOCAL_ONLY_MODE requires LMSTUDIO_BASE_URL to be localhost")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _year_month():
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _get_user(user_id):
    if not user_id:
        return None
    db = _get_db()
    row = db.execute(
        """
        SELECT u.id, u.email, u.is_admin, u.plan_id, p.name AS plan_name,
               p.monthly_char_limit, p.monthly_youtube_minutes_limit
        FROM users u
        LEFT JOIN plans p ON p.id = u.plan_id
        WHERE u.id = ?
        """,
        (user_id,),
    ).fetchone()
    return row


def _get_current_user():
    return _get_user(session.get("user_id"))


def _ensure_guest_user():
    if session.get("user_id"):
        return
    db = _get_db()
    free_plan = db.execute("SELECT id FROM plans WHERE name = 'free'").fetchone()
    email = f"guest-{uuid.uuid4().hex[:10]}@local"
    password_hash = generate_password_hash(uuid.uuid4().hex)
    created_at = _now_iso()
    db.execute(
        "INSERT INTO users (email, password_hash, is_admin, plan_id, created_at) VALUES (?, ?, 0, ?, ?)",
        (email, password_hash, free_plan["id"] if free_plan else None, created_at),
    )
    db.commit()
    session["user_id"] = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()["id"]


@app.before_request
def _attach_user():
    # WHY: ensure every visitor can access pricing/usage flows; WHAT: COMM-001; HOW: remove to require explicit login.
    _ensure_guest_user()
    g.user = _get_current_user()


def _login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not g.user:
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper


def _admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not g.user:
            return redirect(url_for("login", next=request.path))
        if not int(g.user["is_admin"]):
            abort(403)
        return fn(*args, **kwargs)
    return wrapper


def _get_usage(user_id):
    db = _get_db()
    ym = _year_month()
    row = db.execute(
        "SELECT user_id, year_month, chars_used, youtube_minutes_used FROM usage_monthly WHERE user_id=? AND year_month=?",
        (user_id, ym),
    ).fetchone()
    if row:
        return row
    db.execute(
        "INSERT INTO usage_monthly (user_id, year_month, chars_used, youtube_minutes_used, updated_at) VALUES (?, ?, 0, 0, ?)",
        (user_id, ym, _now_iso()),
    )
    db.commit()
    return db.execute(
        "SELECT user_id, year_month, chars_used, youtube_minutes_used FROM usage_monthly WHERE user_id=? AND year_month=?",
        (user_id, ym),
    ).fetchone()


def _check_limits(user_row, chars_to_add=0, youtube_minutes_to_add=0):
    user_id = int(user_row["id"])
    usage = _get_usage(user_id)
    char_limit = user_row["monthly_char_limit"]
    yt_limit = user_row["monthly_youtube_minutes_limit"]

    new_chars = int(usage["chars_used"]) + int(chars_to_add or 0)
    new_yt = int(usage["youtube_minutes_used"]) + int(youtube_minutes_to_add or 0)

    if char_limit is not None and new_chars > int(char_limit):
        raise RuntimeError("Plan limit exceeded: monthly character quota reached")
    if yt_limit is not None and new_yt > int(yt_limit):
        raise RuntimeError("Plan limit exceeded: monthly YouTube minutes quota reached")
    return new_chars, new_yt, usage["year_month"]


def _enforce_and_increment_usage(user_row, chars_to_add=0, youtube_minutes_to_add=0):
    new_chars, new_yt, year_month = _check_limits(user_row, chars_to_add=chars_to_add, youtube_minutes_to_add=youtube_minutes_to_add)
    user_id = int(user_row["id"])
    db = _get_db()
    db.execute(
        "UPDATE usage_monthly SET chars_used=?, youtube_minutes_used=?, updated_at=? WHERE user_id=? AND year_month=?",
        (new_chars, new_yt, _now_iso(), user_id, year_month),
    )
    db.commit()
    return new_chars, new_yt


def _thread_check_and_increment_usage(user_id, chars_to_add=0, youtube_minutes_to_add=0):
    ym = _year_month()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        user = conn.execute(
            """
            SELECT u.id, u.plan_id, p.monthly_char_limit, p.monthly_youtube_minutes_limit
            FROM users u
            LEFT JOIN plans p ON p.id = u.plan_id
            WHERE u.id=?
            """,
            (int(user_id),),
        ).fetchone()
        if not user:
            raise RuntimeError("User not found")
        usage = conn.execute(
            "SELECT user_id, year_month, chars_used, youtube_minutes_used FROM usage_monthly WHERE user_id=? AND year_month=?",
            (int(user_id), ym),
        ).fetchone()
        if not usage:
            conn.execute(
                "INSERT INTO usage_monthly (user_id, year_month, chars_used, youtube_minutes_used, updated_at) VALUES (?, ?, 0, 0, ?)",
                (int(user_id), ym, _now_iso()),
            )
            usage = conn.execute(
                "SELECT user_id, year_month, chars_used, youtube_minutes_used FROM usage_monthly WHERE user_id=? AND year_month=?",
                (int(user_id), ym),
            ).fetchone()
        char_limit = user["monthly_char_limit"]
        yt_limit = user["monthly_youtube_minutes_limit"]
        new_chars = int(usage["chars_used"]) + int(chars_to_add or 0)
        new_yt = int(usage["youtube_minutes_used"]) + int(youtube_minutes_to_add or 0)
        if char_limit is not None and new_chars > int(char_limit):
            raise RuntimeError("Plan limit exceeded: monthly character quota reached")
        if yt_limit is not None and new_yt > int(yt_limit):
            raise RuntimeError("Plan limit exceeded: monthly YouTube minutes quota reached")
        conn.execute(
            "UPDATE usage_monthly SET chars_used=?, youtube_minutes_used=?, updated_at=? WHERE user_id=? AND year_month=?",
            (new_chars, new_yt, _now_iso(), int(user_id), ym),
        )
        conn.commit()
        return new_chars, new_yt
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_init_db()
_ensure_schema()
_ensure_default_plans()
_ensure_default_settings()
_validate_local_backend_urls()

# Load TTS Model (Global)
print("Loading TTS model...")
try:
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID, local_files_only=LOCAL_ONLY_MODE)
    tts_model = VitsModel.from_pretrained(TTS_MODEL_ID, local_files_only=LOCAL_ONLY_MODE)
    print("TTS model loaded.")
except Exception as e:
    print(f"Failed to load TTS model: {e}")
    tts_model = None
    tts_tokenizer = None


def _http_get_json(url):
    response = requests.get(url, timeout=(LLM_CONNECT_TIMEOUT_SECONDS, LLM_READ_TIMEOUT_SECONDS))
    response.raise_for_status()
    return response.json() or {}


def _http_post_json(url, payload):
    response = requests.post(url, json=payload, timeout=(LLM_CONNECT_TIMEOUT_SECONDS, LLM_READ_TIMEOUT_SECONDS))
    response.raise_for_status()
    return response.json() or {}


def _ollama_list_models():
    url = f"{OLLAMA_BASE_URL}/api/tags"
    data = _http_get_json(url)
    models = data.get("models") or []
    names = []
    for m in models:
        name = (m or {}).get("name")
        if name:
            names.append(str(name))
    return names


def _ollama_chat(messages, temperature=0.2, max_tokens=512, model_id=None):
    model = (model_id or OLLAMA_MODEL or "").strip()
    if not model:
        raise RuntimeError("Missing Ollama model name")

    url_chat = f"{OLLAMA_BASE_URL}/api/chat"
    payload_chat = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    last_exception = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = requests.post(url_chat, json=payload_chat, timeout=(LLM_CONNECT_TIMEOUT_SECONDS, LLM_READ_TIMEOUT_SECONDS))
            if response.status_code == 404:
                break
            response.raise_for_status()
            data = response.json() or {}
            msg = (data.get("message") or {})
            content = msg.get("content")
            if not content:
                raise RuntimeError("Empty response from Ollama")
            return str(content)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            last_exception = e
            if attempt >= LLM_MAX_RETRIES:
                raise
            time.sleep(LLM_RETRY_BACKOFF_SECONDS * (attempt + 1))

    url_gen = f"{OLLAMA_BASE_URL}/api/generate"
    system_text = "\n".join([(m.get("content") or "").strip() for m in messages if (m or {}).get("role") == "system"]).strip()
    user_text = "\n".join([(m.get("content") or "").strip() for m in messages if (m or {}).get("role") != "system"]).strip()
    prompt = f"{system_text}\n\n{user_text}".strip()
    payload_gen = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            data = _http_post_json(url_gen, payload_gen)
            content = data.get("response")
            if not content:
                raise RuntimeError("Empty response from Ollama")
            return str(content)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            last_exception = e
            if attempt >= LLM_MAX_RETRIES:
                raise
            time.sleep(LLM_RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise last_exception or RuntimeError("Ollama request failed")


def _lmstudio_list_models():
    url = f"{LMSTUDIO_BASE_URL}/v1/models"
    data = _http_get_json(url)
    models = data.get("data") or []
    model_ids = []
    for m in models:
        mid = (m or {}).get("id")
        if mid:
            model_ids.append(str(mid))
    return model_ids


def _get_lmstudio_model_id(requested_model=None):
    global LMSTUDIO_MODEL_ID

    if requested_model:
        return str(requested_model).strip()

    if LMSTUDIO_MODEL and LMSTUDIO_MODEL.strip().lower() != "auto":
        return LMSTUDIO_MODEL.strip()

    if LMSTUDIO_MODEL_ID:
        return LMSTUDIO_MODEL_ID

    model_ids = _lmstudio_list_models()
    if not model_ids:
        raise RuntimeError("No models available in LM Studio. Start the local server and load gemma 3 4B.")

    preferred = ["google/gemma-3-4b", "google/gemma-3-4b:2", "gemma-kannada-instruct"]
    for p in preferred:
        if p in model_ids:
            LMSTUDIO_MODEL_ID = p
            return p

    LMSTUDIO_MODEL_ID = model_ids[0]
    return model_ids[0]


def _lmstudio_chat(messages, temperature=0.2, max_tokens=512, model_id=None):
    url = f"{LMSTUDIO_BASE_URL}/v1/chat/completions"
    payload = {
        "model": _get_lmstudio_model_id(model_id),
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    last_exception = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            data = _http_post_json(url, payload)
            choices = data.get("choices") or []
            message = (choices[0] or {}).get("message") if choices else None
            content = (message or {}).get("content") if isinstance(message, dict) else None
            if not content:
                raise RuntimeError("Empty response from LM Studio")
            return str(content)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            last_exception = e
            if attempt >= LLM_MAX_RETRIES:
                raise
            time.sleep(LLM_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_exception or RuntimeError("LM Studio request failed")


def _select_llm_backend():
    global LLM_ACTIVE_BACKEND
    if LLM_ACTIVE_BACKEND:
        return LLM_ACTIVE_BACKEND

    if LLM_BACKEND in ("lmstudio", "ollama"):
        LLM_ACTIVE_BACKEND = LLM_BACKEND
        return LLM_ACTIVE_BACKEND

    try:
        _lmstudio_list_models()
        LLM_ACTIVE_BACKEND = "lmstudio"
        return LLM_ACTIVE_BACKEND
    except Exception:
        pass

    try:
        _ollama_list_models()
        LLM_ACTIVE_BACKEND = "ollama"
        return LLM_ACTIVE_BACKEND
    except Exception:
        pass

    LLM_ACTIVE_BACKEND = "lmstudio"
    return LLM_ACTIVE_BACKEND


def _llm_list_models():
    backend = _select_llm_backend()
    if backend == "ollama":
        return _ollama_list_models()
    return _lmstudio_list_models()


def _llm_default_model_id(available_model_ids):
    backend = _select_llm_backend()
    model_ids = available_model_ids or []

    if backend == "ollama":
        preferred = [OLLAMA_MODEL, "gemma3:4b", "gemma:4b", "llama3"]
    else:
        preferred = ["google/gemma-3-4b", "google/gemma-3-4b:2", "gemma-kannada-instruct"]

    for p in preferred:
        if p and p in model_ids:
            return p
    return model_ids[0] if model_ids else None


def _llm_chat(messages, temperature=0.2, max_tokens=512, model_id=None):
    backend = _select_llm_backend()
    if backend == "ollama":
        return _ollama_chat(messages, temperature=temperature, max_tokens=max_tokens, model_id=model_id)
    return _lmstudio_chat(messages, temperature=temperature, max_tokens=max_tokens, model_id=model_id)


def _split_text_for_translation(text, max_chars=2500):
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}".strip()
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        start = 0
        while start < len(p):
            chunks.append(p[start : start + max_chars])
            start += max_chars
    if buf:
        chunks.append(buf)
    return chunks


def _translate_english_to_kannada(text, model_id=None):
    chunks = _split_text_for_translation(text)
    outputs = []
    for chunk in chunks:
        translated = _llm_chat(
            [
                {"role": "system", "content": "Translate English to Kannada. Output only Kannada text."},
                {"role": "user", "content": chunk},
            ],
            temperature=0.1,
            max_tokens=900,
            model_id=model_id,
        )
        outputs.append(translated.strip())
    return "\n\n".join([o for o in outputs if o])


def _ensure_asr_pipeline():
    global ASR_PIPELINE
    with ASR_LOCK:
        if ASR_PIPELINE is not None:
            return ASR_PIPELINE
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_ID,
            model_kwargs={"local_files_only": LOCAL_ONLY_MODE},
            device=device,
        )
        return ASR_PIPELINE


def _safe_job_dir(job_id):
    job_dir = (JOBS_DIR / job_id).resolve()
    if not str(job_dir).startswith(str(JOBS_DIR)):
        raise ValueError("Invalid job path")
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _job_state_path(job_id):
    job_dir = _safe_job_dir(job_id)
    return (job_dir / "job.json").resolve()


def _persist_job_state(job):
    job_id = (job or {}).get("id")
    if not job_id:
        return

    allowed_keys = {
        "id",
        "status",
        "error",
        "url",
        "model_id",
        "video_path",
        "audio_path",
        "subtitles_path",
        "transcript_en_path",
        "transcript_kn_path",
    }
    payload = {k: job.get(k) for k in allowed_keys}
    path = _job_state_path(job_id)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_job_state(job_id):
    try:
        path = _job_state_path(job_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if data.get("id") != job_id:
            data["id"] = job_id
        return data
    except Exception:
        return None


def _rehydrate_jobs_from_disk():
    try:
        for p in JOBS_DIR.iterdir():
            if not p.is_dir():
                continue
            job_id = p.name
            state = _load_job_state(job_id)
            if not state:
                continue
            with JOBS_LOCK:
                JOBS[job_id] = {**state}
    except Exception:
        return


_rehydrate_jobs_from_disk()


def _update_job(job_id, **updates):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            loaded = _load_job_state(job_id)
            if not loaded:
                return
            JOBS[job_id] = loaded
            job = JOBS[job_id]
        job.update(updates)
        _persist_job_state(job)


def _yt_download_best_video(url, out_dir):
    ydl_opts = {
        "outtmpl": str(Path(out_dir) / "%(title).200s-%(id)s.%(ext)s"),
        "format": "best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
        return str(Path(video_path).resolve())


def _yt_extract_audio_wav(url, out_dir):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg to extract audio for transcription.")

    out_dir = Path(out_dir)
    wav_path = (out_dir / "audio.wav").resolve()
    ydl_opts = {
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    if not wav_path.exists():
        candidates = list(out_dir.glob("audio*.wav"))
        if candidates:
            return str(candidates[0].resolve())
        raise RuntimeError("Audio extraction failed: audio.wav not created.")
    return str(wav_path)


def _vtt_to_text(vtt_content):
    lines = []
    for raw in (vtt_content or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("WEBVTT"):
            continue
        if "-->" in line:
            continue
        if line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _parse_vtt_cues(vtt_content):
    cues = []
    lines = (vtt_content or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            times = line.split("-->")
            start = times[0].strip()
            end = times[1].strip()
            i += 1
            text_lines = []
            while i < len(lines):
                t = lines[i].strip()
                if not t:
                    break
                if "-->" in t:
                    i -= 1
                    break
                if not t.isdigit() and not t.upper().startswith("WEBVTT"):
                    text_lines.append(re.sub(r"<[^>]+>", "", t))
                i += 1
            text = " ".join(text_lines).strip()
            if text:
                cues.append({"start": start, "end": end, "text_en": text})
        i += 1
    return cues

def _build_vtt_from_cues(cues, text_field):
    out = ["WEBVTT", ""]
    idx = 1
    for cue in cues:
        t = (cue.get(text_field) or "").strip()
        if not t:
            continue
        out.append(str(idx))
        out.append(f"{cue['start']} --> {cue['end']}")
        out.append(t)
        out.append("")
        idx += 1
    return "\n".join(out)


def _yt_download_english_subtitles(url, out_dir):
    out_dir = Path(out_dir)
    ydl_opts = {
        "outtmpl": str(out_dir / "subs.%(ext)s"),
        "skip_download": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    candidates = list(out_dir.glob("subs*.vtt"))
    if not candidates:
        return None, None
    vtt_path = candidates[0].resolve()
    vtt_text = vtt_path.read_text(encoding="utf-8", errors="ignore")
    transcript = _vtt_to_text(vtt_text)
    if not transcript:
        return str(vtt_path), None
    return str(vtt_path), transcript

def _extract_youtube_video_id(url):
    try:
        # Common patterns: watch?v=ID, youtu.be/ID, embed/ID
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
        if m:
            return m.group(1)
        m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
        if m:
            return m.group(1)
        m = re.search(r"embed/([A-Za-z0-9_-]{11})", url)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None

def _sec_to_vtt_time(sec):
    s = float(sec or 0.0)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    seconds = s % 60
    return f"{h:02d}:{m:02d}:{int(seconds):02d}.{int((seconds-int(seconds))*1000):03d}"

def _yt_fetch_transcript_api(url, out_dir):
    if YouTubeTranscriptApi is None:
        return None, None
    vid = _extract_youtube_video_id(url)
    if not vid:
        return None, None
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
        # Try English variants
        for lang in ["en", "en-US", "en-GB"]:
            try:
                t = transcripts.find_transcript([lang])
                items = t.fetch()
                job_dir = Path(out_dir)
                vtt_path = (job_dir / "subs.en.vtt").resolve()
                lines = ["WEBVTT", ""]
                text_lines = []
                idx = 1
                for it in items:
                    start = _sec_to_vtt_time(it.get("start", 0))
                    end = _sec_to_vtt_time(it.get("start", 0) + it.get("duration", 0))
                    txt = (it.get("text") or "").replace("\n", " ").strip()
                    if not txt:
                        continue
                    lines.append(str(idx))
                    lines.append(f"{start} --> {end}")
                    lines.append(txt)
                    lines.append("")
                    idx += 1
                    text_lines.append(txt)
                vtt_path.write_text("\n".join(lines), encoding="utf-8")
                transcript_text = " ".join(text_lines)
                return str(vtt_path), transcript_text
            except (NoTranscriptFound, TranscriptsDisabled):
                continue
        # Try auto-generated if manual is not found
        try:
            t = transcripts.find_generated_transcript(["en", "en-US", "en-GB"])
            items = t.fetch()
            job_dir = Path(out_dir)
            vtt_path = (job_dir / "subs.en.vtt").resolve()
            lines = ["WEBVTT", ""]
            text_lines = []
            idx = 1
            for it in items:
                start = _sec_to_vtt_time(it.get("start", 0))
                end = _sec_to_vtt_time(it.get("start", 0) + it.get("duration", 0))
                txt = (it.get("text") or "").replace("\n", " ").strip()
                if not txt:
                    continue
                lines.append(str(idx))
                lines.append(f"{start} --> {end}")
                lines.append(txt)
                lines.append("")
                idx += 1
                text_lines.append(txt)
            vtt_path.write_text("\n".join(lines), encoding="utf-8")
            transcript_text = " ".join(text_lines)
            return str(vtt_path), transcript_text
        except Exception:
            return None, None
    except Exception:
        return None, None
    return None, None


def _transcribe_english(audio_path):
    asr = _ensure_asr_pipeline()
    result = asr(audio_path, chunk_length_s=30, batch_size=4)
    text = result.get("text") if isinstance(result, dict) else str(result)
    return (text or "").strip()


def _run_youtube_job(job_id, url):
    try:
        job_dir = _safe_job_dir(job_id)
        _update_job(job_id, status="downloading_video", error=None)
        video_path = _yt_download_best_video(url, job_dir)
        _update_job(job_id, video_path=video_path)

        _update_job(job_id, status="fetching_transcript")
        sub_path, transcript_en = _yt_download_english_subtitles(url, job_dir)
        _update_job(job_id, subtitles_path=sub_path)

        if not transcript_en:
            _update_job(job_id, status="fetching_transcript_api")
            sp2, transcript_api = _yt_fetch_transcript_api(url, job_dir)
            if transcript_api:
                _update_job(job_id, subtitles_path=sp2)
                transcript_en = transcript_api
            else:
                if shutil.which("ffmpeg") is None:
                    raise RuntimeError(
                        "No English subtitles found, and ffmpeg is not installed to transcribe audio. Install ffmpeg or ensure English captions are enabled on the video."
                    )

            _update_job(job_id, status="extracting_audio")
            audio_path = _yt_extract_audio_wav(url, job_dir)
            _update_job(job_id, audio_path=audio_path)

            _update_job(job_id, status="transcribing_audio")
            transcript_en = _transcribe_english(audio_path)

        en_path = (job_dir / "transcript_en.txt").resolve()
        en_path.write_text(transcript_en, encoding="utf-8")
        _update_job(job_id, transcript_en=transcript_en, transcript_en_path=str(en_path))

        _update_job(job_id, status="translating_to_kannada")
        with JOBS_LOCK:
            job = JOBS.get(job_id) or {}
            model_id = job.get("model_id")
        transcript_kn = _translate_english_to_kannada(transcript_en, model_id=model_id)
        kn_path = (job_dir / "transcript_kn.txt").resolve()
        kn_path.write_text(transcript_kn, encoding="utf-8")
        _update_job(job_id, transcript_kn=transcript_kn, transcript_kn_path=str(kn_path))

        _update_job(job_id, status="completed")
    except Exception as e:
        _update_job(job_id, status="error", error=str(e))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/api/user/usage")
@_login_required
def get_user_usage():
    user = g.user
    usage = _get_usage(user["id"])
    return jsonify({
        "email": user["email"],
        "plan": user["plan_name"],
        "is_admin": bool(user["is_admin"]),
        "chars_used": usage["chars_used"],
        "chars_limit": user["monthly_char_limit"],
        "youtube_minutes_used": usage["youtube_minutes_used"],
        "youtube_minutes_limit": user["monthly_youtube_minutes_limit"]
    })


@app.route("/api/plans")
def get_plans():
    db = _get_db()
    rows = db.execute(
        """
        SELECT id, name, monthly_char_limit, monthly_youtube_minutes_limit, price_cents, summary, features_json
        FROM plans
        WHERE is_active=1
        """
    ).fetchall()
    plans = []
    for row in rows:
        plan = dict(row)
        try:
            plan["features"] = json.loads(plan.get("features_json") or "[]")
        except json.JSONDecodeError:
            plan["features"] = []
        plans.append(plan)
    return jsonify(plans)


@app.route("/api/system/status", methods=["GET"])
def system_status():
    # WHY: expose local-only + TTS state for UI; WHAT: COMM-002; HOW: remove endpoint and banner to roll back.
    return jsonify(
        {
            "local_only_mode": _get_setting("local_only_mode", "0") == "1",
            "tts_available": bool(tts_model),
            "asr_model_id": ASR_MODEL_ID,
        }
    )


@app.route("/api/subscription", methods=["GET"])
def subscription_status():
    user = g.user
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(
        {
            "user_id": user["id"],
            "plan_id": user["plan_id"],
            "plan_name": user["plan_name"],
        }
    )


@app.route("/api/subscribe", methods=["POST"])
def subscribe():
    user = g.user
    if not user:
        return jsonify({"error": "User not found"}), 404
    data = request.json or {}
    plan_name = (data.get("plan_name") or "").strip().lower()
    if not plan_name:
        return jsonify({"error": "Missing plan_name"}), 400
    db = _get_db()
    plan = db.execute(
        "SELECT id, name FROM plans WHERE name = ? AND is_active = 1",
        (plan_name,),
    ).fetchone()
    if not plan:
        return jsonify({"error": "Plan not available"}), 404
    # WHY: enable self-serve upgrades; WHAT: COMM-001; HOW: disable by removing this update and endpoint.
    db.execute("UPDATE users SET plan_id=? WHERE id=?", (plan["id"], user["id"]))
    db.commit()
    return jsonify({"plan_id": plan["id"], "plan_name": plan["name"]})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tts', methods=['POST'])
def tts():
    if not tts_model:
        return jsonify({'error': 'TTS model not loaded'}), 503
        
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
        
    try:
        inputs = tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
            
        waveform = output.cpu().numpy().squeeze()
        
        # Create in-memory file
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, tts_model.config.sampling_rate, waveform)
        buffer.seek(0)
        
        return send_file(buffer, mimetype="audio/wav")
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get('input_text', '')
    mode = data.get('mode', 'chat') # chat, to_kannada, to_english
    model_id = (data.get("model") or "").strip() or None

    if not user_input:
        return jsonify({'error': 'No input text provided'}), 400

    system_prompt = "You are a helpful AI assistant."
    user_prompt = user_input

    if mode == 'to_kannada':
        system_prompt = "Translate English to Kannada. Output only Kannada text."
        user_prompt = user_input
    elif mode == 'to_english':
        system_prompt = "Translate Kannada to English. Output only English text."
        user_prompt = user_input
    elif mode == 'chat':
        system_prompt = "You are a helpful assistant fluent in Kannada and English."

    try:
        ai_response = _llm_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4 if mode == "chat" else 0.1,
            max_tokens=700 if mode == "chat" else 900,
            model_id=model_id,
        )
        
        return jsonify({'response': ai_response})
    
    except requests.exceptions.ReadTimeout:
        return jsonify({'error': f"API Error: LLM timed out after {LLM_READ_TIMEOUT_SECONDS}s. Try again."}), 504
    except requests.exceptions.ConnectTimeout:
        return jsonify({'error': f"API Error: Could not connect to LLM within {LLM_CONNECT_TIMEOUT_SECONDS}s."}), 504
    except requests.exceptions.ConnectionError as e:
        return jsonify({'error': f"API Error: Could not reach LLM: {str(e)}"}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"API Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({'error': f"Server Error: {str(e)}"}), 500


@app.route('/api/youtube/start', methods=['POST'])
def youtube_start():
    data = request.json or {}
    url = (data.get("url") or "").strip()
    model_id = (data.get("model") or "").strip() or None
    if not url:
        return jsonify({"error": "Missing YouTube URL"}), 400

    job_id = uuid.uuid4().hex
    user_id = (g.user["id"] if g.user else None)
    job = {
        "id": job_id,
        "status": "queued",
        "error": None,
        "url": url,
        "model_id": model_id,
        "user_id": user_id,
        "video_path": None,
        "audio_path": None,
        "subtitles_path": None,
        "transcript_en_path": None,
        "transcript_kn_path": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    _persist_job_state(job)

    thread = threading.Thread(target=_run_youtube_job, args=(job_id, url), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.route('/api/youtube/status/<job_id>', methods=['GET'])
def youtube_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            loaded = _load_job_state(job_id)
            if not loaded:
                return jsonify({"error": "Job not found"}), 404
            JOBS[job_id] = loaded
            job = loaded

        def _exists(path):
            return bool(path) and Path(path).exists()

        return jsonify(
            {
                "id": job_id,
                "status": job.get("status"),
                "error": job.get("error"),
                "model_id": job.get("model_id"),
                "video_ready": _exists(job.get("video_path")),
                "subtitles_ready": _exists(job.get("subtitles_path")),
                "subtitles_bilingual_ready": _exists(job.get("subtitles_bilingual_path")),
                "transcript_en_ready": _exists(job.get("transcript_en_path")),
                "transcript_kn_ready": _exists(job.get("transcript_kn_path")),
            }
        )


@app.route('/api/models', methods=['GET'])
def api_models():
    try:
        backend = (request.args.get("backend") or "").strip().lower()
        if backend == "ollama":
            model_ids = _ollama_list_models()
            return jsonify({"models": model_ids, "default": _llm_default_model_id(model_ids), "backend": "ollama"})
        if backend == "lmstudio":
            model_ids = _lmstudio_list_models()
            return jsonify({"models": model_ids, "default": _llm_default_model_id(model_ids), "backend": "lmstudio"})
        model_ids = _llm_list_models()
        return jsonify({"models": model_ids, "default": _llm_default_model_id(model_ids), "backend": _select_llm_backend()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download/<job_id>/<artifact>', methods=['GET'])
def download_artifact(job_id, artifact):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

    mapping = {
        "video": job.get("video_path"),
        "transcript_en": job.get("transcript_en_path"),
        "transcript_kn": job.get("transcript_kn_path"),
        "subtitles": job.get("subtitles_path"),
        "subtitles_bilingual": job.get("subtitles_bilingual_path"),
    }
    path = mapping.get(artifact)
    if not path:
        return jsonify({"error": "Artifact not found"}), 404

    file_path = Path(path).resolve()
    job_dir = _safe_job_dir(job_id)
    if not str(file_path).startswith(str(job_dir)):
        return jsonify({"error": "Invalid artifact path"}), 400
    if not file_path.exists():
        return jsonify({"error": "File not ready"}), 404

    return send_file(str(file_path), as_attachment=True)

@app.route('/stream/<job_id>/<artifact>', methods=['GET'])
def stream_artifact(job_id, artifact):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

    mapping = {
        "video": job.get("video_path"),
        "subtitles": job.get("subtitles_path"),
        "subtitles_bilingual": job.get("subtitles_bilingual_path"),
    }
    path = mapping.get(artifact)
    if not path:
        return jsonify({"error": "Artifact not found"}), 404

    file_path = Path(path).resolve()
    job_dir = _safe_job_dir(job_id)
    if not str(file_path).startswith(str(job_dir)):
        return jsonify({"error": "Invalid artifact path"}), 400
    if not file_path.exists():
        return jsonify({"error": "File not ready"}), 404

    if artifact == "subtitles":
        return send_file(str(file_path), as_attachment=False, mimetype="text/vtt")
    if artifact == "subtitles_bilingual":
        return send_file(str(file_path), as_attachment=False, mimetype="text/vtt")
    return send_file(str(file_path), as_attachment=False)

@app.route('/api/youtube/cues/<job_id>', methods=['GET'])
def youtube_cues(job_id):
    model_id = (request.args.get("model") or "").strip() or None
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            loaded = _load_job_state(job_id)
            if not loaded:
                return jsonify({"error": "Job not found"}), 404
            JOBS[job_id] = loaded
            job = loaded
    sub_path = job.get("subtitles_path")
    if not sub_path or not Path(sub_path).exists():
        return jsonify({"error": "Subtitles not ready"}), 400
    if job.get("subtitles_bilingual_path") and Path(job.get("subtitles_bilingual_path")).exists():
        cues_json_path = Path(_safe_job_dir(job_id)) / "cues.json"
        if cues_json_path.exists():
            try:
                cues = json.loads(cues_json_path.read_text(encoding="utf-8"))
            except Exception:
                cues = []
        else:
            cues = []
        return jsonify({"cues": cues, "bilingual_ready": True})
    lock = BILINGUAL_LOCKS.get(job_id)
    if lock is None:
        lock = threading.Lock()
        BILINGUAL_LOCKS[job_id] = lock
    if not lock.acquire(blocking=False):
        return jsonify({"status": "generating", "message": "Bilingual subtitles generation in progress"}), 202
    _update_job(job_id, status="generating_bilingual_subtitles")
    vtt_text = Path(sub_path).read_text(encoding="utf-8", errors="ignore")
    cues = _parse_vtt_cues(vtt_text)
    outputs = []
    total_est_chars = sum(len((c.get("text_en") or "")) for c in cues)
    total_est_chars = int(total_est_chars * 1.3)
    user_row = _get_current_user()
    if user_row:
        try:
            _check_limits(user_row, chars_to_add=total_est_chars)
        except Exception as e:
            lock.release()
            return jsonify({"error": str(e)}), 403
    for cue in cues:
        en = cue.get("text_en") or ""
        mix = _llm_chat(
            [
                {"role": "system", "content": "Translate the text to Kannada and include about 20% of the original English to preserve flow. Keep a single concise line with Kannada first, then minimal English hints in brackets."},
                {"role": "user", "content": en},
            ],
            temperature=0.1,
            max_tokens=120,
            model_id=model_id,
        )
        out = {**cue, "text_kn_mix": mix.strip()}
        outputs.append(out)
        # small sleep to avoid overwhelming local LLM when large subtitle files
        time.sleep(0.01)
    total_actual_chars = sum(len((o.get("text_kn_mix") or "")) for o in outputs)
    try:
        if user_row:
            _enforce_and_increment_usage(user_row, chars_to_add=total_actual_chars)
    except Exception as e:
        lock.release()
        return jsonify({"error": str(e)}), 403
    job_dir = _safe_job_dir(job_id)
    cues_json_path = (job_dir / "cues.json").resolve()
    cues_json_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    bil_vtt = _build_vtt_from_cues(outputs, "text_kn_mix")
    bil_vtt_path = (job_dir / "subs.bilingual.vtt").resolve()
    bil_vtt_path.write_text(bil_vtt, encoding="utf-8")
    _update_job(job_id, subtitles_bilingual_path=str(bil_vtt_path), status="completed" if (JOBS.get(job_id) or {}).get("status") == "generating_bilingual_subtitles" else (JOBS.get(job_id) or {}).get("status"))
    lock.release()
    return jsonify({"cues": outputs, "bilingual_ready": True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
