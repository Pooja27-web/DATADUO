import os
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

load_dotenv()

_RETRY_CODES = (429, 500, 502, 503, 504)
LAST_PROVIDER = {"name": None}


def _label(base_url: str, model: str) -> str:
    host = (urlparse(base_url).hostname or "").lower()
    names = [
        ("generativelanguage", "Gemini"),
        ("groq", "Groq"),
        ("openrouter", "OpenRouter"),
        ("localhost", "Ollama (local)"),
        ("127.0.0.1", "Ollama (local)"),
    ]
    base = next((n for key, n in names if key in host), host or "LLM")
    return f"{base} · {model}"


def _load_providers():
    # Slot 1 = your original three variables; slots 2-6 = extra AIs
    slots = [("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")]
    slots += [(f"LLM{i}_API_KEY", f"LLM{i}_BASE_URL", f"LLM{i}_MODEL") for i in range(2, 7)]
    found = []
    for k, b, m in slots:
        key, base, model = os.getenv(k), os.getenv(b), os.getenv(m)
        if key and base and model:
            found.append({
                "name": _label(base, model),
                "model": model,
                "client": OpenAI(api_key=key, base_url=base, timeout=25, max_retries=0),
            })
    return found


_PROVIDERS = _load_providers()


def available_providers():
    return [p["name"] for p in _PROVIDERS]


def _order(preferred):
    if preferred:
        first = [p for p in _PROVIDERS if p["name"] == preferred]
        rest = [p for p in _PROVIDERS if p["name"] != preferred]
        return first + rest
    return list(_PROVIDERS)


def _messages(prompt, system):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def _check():
    if not _PROVIDERS:
        raise RuntimeError(
            "No AI is configured. Add LLM_API_KEY, LLM_BASE_URL and LLM_MODEL to your .env file."
        )


def ask(prompt: str, system: str = "You are a helpful data analyst.", provider=None) -> str:
    """Ask the AI. Fails over to the next provider if one is slow or down."""
    _check()
    last = None
    for p in _order(provider):
        for _ in range(2):
            try:
                r = p["client"].chat.completions.create(
                    model=p["model"], messages=_messages(prompt, system)
                )
                LAST_PROVIDER["name"] = p["name"]
                return r.choices[0].message.content
            except APIConnectionError as e:
                last = e
            except APIStatusError as e:
                last = e
                if e.status_code not in _RETRY_CODES:
                    break  # bad key / bad model name: try the next provider
            time.sleep(1)
    raise RuntimeError("All AI providers are busy or unreachable. Try again in a minute.") from last


def ask_stream(prompt: str, system: str = "You are a helpful data analyst.", provider=None):
    """Same as ask(), but yields the answer piece by piece (feels much faster)."""
    _check()
    last = None
    for p in _order(provider):
        started = False
        try:
            stream = p["client"].chat.completions.create(
                model=p["model"], messages=_messages(prompt, system), stream=True
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    started = True
                    LAST_PROVIDER["name"] = p["name"]
                    yield piece
            if started:
                return
        except Exception as e:
            last = e
            if started:
                raise
    raise RuntimeError("All AI providers are busy or unreachable. Try again in a minute.") from last