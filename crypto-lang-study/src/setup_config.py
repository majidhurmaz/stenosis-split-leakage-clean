"""Part C, step 2: write the real model list into config.yaml from what Ollama has installed.

- asks the running Ollama server which models are installed (GET /api/tags)
- for each study model, picks the exact installed tag (or a close match, with a warning)
- removes the mock model, keeps any commercial (openai_compat) models you added yourself
- saves the old file as config.yaml.bak
- writes results/model_versions.json: exact tag, digest, size, quantisation, date -> Table 2

Usage:  python src/setup_config.py            (normally called by partC.ps1)
        python src/setup_config.py --keep-mock
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys

import requests
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_config, out_dir  # noqa: E402

# study id -> (preferred tag, base name used for a fallback match)
STUDY_MODELS = [
    ("qwen2.5-coder-7b", "qwen2.5-coder:7b-instruct-q4_K_M", "qwen2.5-coder:7b"),
    ("llama3.1-8b", "llama3.1:8b-instruct-q4_K_M", "llama3.1:8b"),
    ("deepseek-coder-6.7b", "deepseek-coder:6.7b-instruct-q4_K_M", "deepseek-coder:6.7b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-mock", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    url = cfg.get("ollama_url", "http://localhost:11434")
    try:
        tags = requests.get(f"{url}/api/tags", timeout=10).json().get("models", [])
    except requests.RequestException:
        sys.exit(f"Ollama is not answering at {url}. Start the Ollama app and run this again.")
    installed = {m["name"]: m for m in tags}
    print("Installed in Ollama:", ", ".join(installed) or "(nothing)")

    models, versions, missing = [], [], []
    for sid, preferred, base in STUDY_MODELS:
        tag = preferred if preferred in installed else next(
            (n for n in installed if n.startswith(base)), None)
        if tag is None:
            missing.append(preferred)
            continue
        if tag != preferred:
            print(f"  NOTE: {sid}: using '{tag}' (preferred '{preferred}' not installed)")
        models.append({"id": sid, "backend": "ollama", "name": tag})
        m = installed[tag]
        det = m.get("details", {})
        versions.append({"id": sid, "tag": tag, "digest": m.get("digest"), "size_bytes": m.get("size"),
                         "parameter_size": det.get("parameter_size"), "quantization": det.get("quantization_level"),
                         "family": det.get("family"), "ollama_modified_at": m.get("modified_at")})

    # keep the user's own commercial models; drop mock unless asked
    for m in cfg.get("models", []):
        if m.get("backend") == "openai_compat" or (args.keep_mock and m.get("backend") == "mock"):
            models.append(m)

    if not models:
        sys.exit("No study model is installed yet. Run the model downloads first (partC.ps1 does this).")

    path = ROOT / "config.yaml"
    shutil.copy(path, ROOT / "config.yaml.bak")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["models"] = models
    header = ("# Study configuration. Model list written by src/setup_config.py on "
              f"{dt.date.today().isoformat()} from the models installed in Ollama.\n"
              "# Previous version: config.yaml.bak. To add a commercial model see README.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)

    try:
        ver = requests.get(f"{url}/api/version", timeout=10).json().get("version")
    except requests.RequestException:
        ver = None
    out = out_dir(cfg, ".")
    with open(out / "model_versions.json", "w", encoding="utf-8") as f:
        json.dump({"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "ollama_version": ver,
                   "models": versions}, f, indent=2)

    print(f"\nconfig.yaml now has {len(models)} model(s): {', '.join(m['id'] for m in models)}")
    print(f"Model versions for Table 2: {out / 'model_versions.json'}")
    if missing:
        print("STILL MISSING (download did not finish):", ", ".join(missing))
        sys.exit(2)


if __name__ == "__main__":
    main()
