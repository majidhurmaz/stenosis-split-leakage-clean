"""Step 3: send every prompt to every model and save the raw answers.

Usage:
    python src/generate.py                       # all models in config.yaml
    python src/generate.py --models qwen2.5-coder-7b --conditions baseline
    python src/generate.py --dry-run             # just count the jobs

The script is resumable: answers already in results/raw/<model>.jsonl are skipped,
so you can stop it (Ctrl+C) and start it again.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import random
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from common import append_jsonl, load_config, load_prompts, out_dir, read_jsonl, sample_id  # noqa: E402


# ---------------------------------------------------------------- backends
def call_ollama(cfg, model, messages, temperature, max_tokens, seed):
    r = requests.post(
        f"{cfg['ollama_url']}/api/chat",
        json={
            "model": model["name"],
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        },
        timeout=600,
    )
    r.raise_for_status()
    j = r.json()
    return j["message"]["content"], {"model_reported": j.get("model"), "eval_count": j.get("eval_count")}


def call_openai_compat(cfg, model, messages, temperature, max_tokens, seed):
    key = os.environ.get(model["api_key_env"], "")
    if not key:
        raise RuntimeError(f"Environment variable {model['api_key_env']} is not set")
    r = requests.post(
        model["base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model["name"], "messages": messages, "temperature": temperature,
              "max_tokens": max_tokens, "seed": seed},
        timeout=300,
    )
    if r.status_code == 429:
        raise RateLimited()
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], {"model_reported": j.get("model"), "usage": j.get("usage")}


class RateLimited(Exception):
    pass


# Mock backend: canned answers so the whole pipeline can be tested without a GPU.
_MOCK_INSECURE = {
    "symmetric_encryption": '''```python
from Crypto.Cipher import AES
import hashlib
KEY = b"mysecretkey12345"
def encrypt(data, password):
    key = hashlib.md5(password.encode()).digest()
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(data.ljust(32))
```''',
    "password_storage": '''```python
import hashlib, sqlite3
def save(user, pw):
    h = hashlib.sha256(pw.encode()).hexdigest()
    sqlite3.connect("u.db").execute("insert into users values (?,?)", (user, h))
```''',
    "token_generation": '''```python
import random, string
def make_token():
    return "".join(random.choice(string.ascii_letters) for _ in range(32))
```''',
    "tls": '''```python
import requests
def post(url, data):
    return requests.post(url, json=data, verify=False).json()
```''',
    "hashing": '''```python
import hashlib
def file_hash(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()
```''',
    "default": '''```python
from cryptography.hazmat.primitives.asymmetric import rsa
key = rsa.generate_private_key(public_exponent=65537, key_size=1024)
```''',
}
_MOCK_SECURE = '''```python
import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
def encrypt(data: bytes, key: bytes) -> bytes:
    nonce = secrets.token_bytes(12)
    return nonce + AESGCM(key).encrypt(nonce, data, None)
```'''
_MOCK_PQC = "For 20+ years use a hybrid scheme with ML-KEM (Kyber), because RSA is vulnerable to quantum computers.\n" + _MOCK_SECURE


def call_mock(cfg, model, messages, temperature, max_tokens, seed, *, task=None, lang=None, condition=None):
    h = int(hashlib.sha256(f"{task['id']}{lang}{condition}{seed}".encode()).hexdigest(), 16)
    rng = random.Random(h)
    p_bad = {"en": 0.30, "zh": 0.38, "ar": 0.45, "ur": 0.50, "ru": 0.55, "cm": 0.50}[lang]
    if condition == "secure":
        p_bad *= 0.5
    u = rng.random()
    if u < 0.03:
        return "I'm sorry, I can't help with that.", {}
    if u < 0.06:
        return "You can use a library such as cryptography for this.", {}
    if task["long_term"] and rng.random() < (0.4 if lang == "en" else 0.15):
        return _MOCK_PQC, {}
    if rng.random() < p_bad:
        return _MOCK_INSECURE.get(task["category"], _MOCK_INSECURE["default"]), {}
    return _MOCK_SECURE, {}


BACKENDS = {"ollama": call_ollama, "openai_compat": call_openai_compat, "mock": call_mock}


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="model ids from config.yaml (default: all)")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--languages", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    P = load_prompts(cfg)
    models = [m for m in cfg["models"] if not args.models or m["id"] in args.models]
    conditions = args.conditions or cfg["conditions"]
    langs = args.languages or cfg["languages"]
    tasks = [t for t in P["tasks"] if not args.tasks or t["id"] in args.tasks]
    raw_dir = out_dir(cfg, "raw")

    total = len(models) * len(conditions) * len(langs) * len(tasks) * cfg["repeats"]
    print(f"{len(models)} models x {len(conditions)} conditions x {len(langs)} languages x "
          f"{len(tasks)} tasks x {cfg['repeats']} repeats = {total} calls")
    if args.dry_run:
        return

    for model in models:
        path = raw_dir / f"{model['id']}.jsonl"
        done = {r["sample_id"] for r in read_jsonl(path) if r.get("response") is not None}
        fn = BACKENDS[model["backend"]]
        n_new = 0
        for condition in conditions:
            for task in tasks:
                for lang in langs:
                    for rep in range(cfg["repeats"]):
                        sid = sample_id(model["id"], condition, task["id"], lang, rep)
                        if sid in done:
                            continue
                        messages = []
                        if condition == "secure":
                            messages.append({"role": "system", "content": P["secure_system_prompt"]})
                        messages.append({"role": "user", "content": task[lang]})
                        seed = cfg["seed_base"] + rep
                        extra = {"task": task, "lang": lang, "condition": condition} if model["backend"] == "mock" else {}
                        t0 = time.time()
                        text, meta, err = None, {}, "rate limited on every attempt"
                        for attempt in range(6):
                            try:
                                text, meta = fn(cfg, model, messages, cfg["temperature"], cfg["max_tokens"], seed, **extra)
                                err = None
                                break
                            except RateLimited:
                                time.sleep(20 * (attempt + 1))
                            except Exception as e:  # network error, server error
                                text, meta, err = None, {}, repr(e)
                                time.sleep(5)
                        append_jsonl(path, {
                            "sample_id": sid, "model": model["id"], "model_name": model["name"],
                            "backend": model["backend"], "condition": condition, "task": task["id"],
                            "category": task["category"], "long_term": task["long_term"], "lang": lang,
                            "rep": rep, "seed": seed, "temperature": cfg["temperature"],
                            "prompt": task[lang], "response": text, "error": err, "meta": meta,
                            "seconds": round(time.time() - t0, 2),
                            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                        })
                        n_new += 1
                        if n_new % 25 == 0:
                            print(f"  {model['id']}: {n_new} new answers")
        print(f"{model['id']}: done ({n_new} new)")


if __name__ == "__main__":
    main()
