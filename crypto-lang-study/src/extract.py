"""Step 4: pull the Python code out of every answer and classify the answer.

Output:
    results/code/<sample_id>.py   one file per answer that contains code
    results/samples.csv           one row per answer: status, parses, n_lines ...

status: code | no_code | refusal | error
"""
from __future__ import annotations

import ast
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import load_config, out_dir, read_jsonl  # noqa: E402

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)
NON_PY = {"bash", "sh", "shell", "console", "cmd", "powershell", "ps1", "text", "txt", "json",
          "yaml", "yml", "ini", "toml", "html", "sql", "javascript", "js", "env", "dockerfile", "plaintext"}
CODE_HINT = re.compile(r"^\s*(import |from \w+ import |def |class |@app\.)", re.M)

REFUSAL = re.compile("|".join([
    r"\bI(?:'m| am)? (?:sorry|unable)\b", r"\bI can(?:not|'t) (?:help|assist|provide)",
    r"معذرت", r"معاف(?:ی)? (?:چاہتا|کیجئے)", r"\bmaazrat\b", r"\bmaaf(?:i)? (?:kijiye|karna)\b",
    r"nahi?n? kar sakta", r"抱歉", r"无法(?:提供|协助|帮助)", r"عذراً", r"لا أستطيع", r"لا يمكنني",
]), re.I)


def extract_code(text: str) -> str:
    blocks = []
    for lang, body in FENCE.findall(text):
        lang = lang.lower()
        if lang in NON_PY:
            continue
        if lang in ("python", "py", "python3") or CODE_HINT.search(body):
            blocks.append(body.strip("\n"))
    if not blocks and CODE_HINT.search(text) and "```" not in text:
        blocks = [text]  # unfenced answer that is just code
    return "\n\n".join(blocks)


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def main():
    cfg = load_config()
    raw_dir = out_dir(cfg, "raw")
    code_dir = out_dir(cfg, "code")
    rows = {}
    for f in sorted(raw_dir.glob("*.jsonl")):
        for r in read_jsonl(f):
            prev = rows.get(r["sample_id"])
            if prev is None or (prev.get("response") is None and r.get("response") is not None):
                rows[r["sample_id"]] = r  # keep the successful attempt if there were retries

    out = []
    for sid, r in rows.items():
        text = r.get("response")
        code = ""
        if text is None:
            status = "error"
        else:
            code = extract_code(text)
            if code.strip():
                status = "code"
            elif REFUSAL.search(text):
                status = "refusal"
            else:
                status = "no_code"
        ok = parses(code) if code else False
        if code:
            (code_dir / f"{sid}.py").write_text(code, encoding="utf-8")
        out.append({k: r[k] for k in ("sample_id", "model", "model_name", "condition", "task", "category",
                                       "long_term", "lang", "rep", "timestamp")}
                   | {"status": status, "parses": ok, "n_lines": code.count("\n") + 1 if code else 0,
                      "response_chars": len(text or "")})
    df = pd.DataFrame(out).sort_values("sample_id")
    df.to_csv(cfg["_out"] / "samples.csv", index=False, encoding="utf-8")
    print(df.groupby(["lang", "status"]).size().unstack(fill_value=0))
    print(f"\n{(df.status == 'code').sum()} code samples, {(~df.parses & (df.status == 'code')).sum()} do not parse")


if __name__ == "__main__":
    main()
