"""Shared helpers: config loading, paths, JSONL I/O."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Windows consoles/log files may not be UTF-8: never crash on printing Urdu, Arabic or Chinese.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_out"] = ROOT / cfg.get("output_dir", "results")
    return cfg


def load_prompts(cfg: dict) -> dict:
    with open(ROOT / cfg["prompts_file"], encoding="utf-8") as f:
        return json.load(f)


def out_dir(cfg: dict, *parts: str) -> Path:
    p = cfg["_out"].joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_id(model: str, condition: str, task: str, lang: str, rep: int) -> str:
    return f"{model}__{condition}__{task}__{lang}__r{rep}"
