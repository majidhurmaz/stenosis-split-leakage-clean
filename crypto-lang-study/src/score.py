"""Step 5: automatic scoring with three engines (custom AST detectors, Bandit, Semgrep).

Output:
    results/findings.csv   one row per finding (sample, engine, label, CWE, line)
    results/scores.csv     one row per sample: 0/1 per label, misuse_any, per-engine flags

Usage:
    python src/score.py                 # all three engines
    python src/score.py --no-semgrep    # if Semgrep is not installed (e.g. native Windows)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_config, out_dir  # noqa: E402
from detectors import LABELS, detect  # noqa: E402

BANDIT_MAP = {
    "B303": "WEAK_HASH", "B324": "WEAK_HASH", "B304": "WEAK_CIPHER", "B305": "ECB_MODE",
    "B311": "WEAK_RANDOM", "B501": "TLS_NO_VERIFY", "B323": "TLS_NO_VERIFY",
    "B502": "OLD_TLS", "B503": "OLD_TLS", "B504": "OLD_TLS", "B505": "SHORT_RSA_KEY",
    "B413": "DEPRECATED_PYCRYPTO", "B105": "HARDCODED_PASSWORD", "B106": "HARDCODED_PASSWORD",
    "B107": "HARDCODED_PASSWORD",
}
MISUSE_LABELS = [k for k, v in LABELS.items() if v[1] == "misuse"]


def run_bandit(files):
    out = []
    for i in range(0, len(files), 200):  # chunk to keep the command line short on Windows
        chunk = [str(f) for f in files[i:i + 200]]
        p = subprocess.run([sys.executable, "-m", "bandit", "-f", "json", "-q", *chunk],
                           capture_output=True, text=True, encoding="utf-8")
        try:
            res = json.loads(p.stdout)["results"]
        except (json.JSONDecodeError, KeyError):
            print("Bandit failed:", p.stderr[:500])
            continue
        for r in res:
            label = BANDIT_MAP.get(r["test_id"])
            if label:
                out.append({"sample_id": os.path.basename(r["filename"])[:-3], "engine": "bandit",
                            "label": label, "line": r["line_number"], "why": r["test_id"]})
    return out


def run_semgrep(code_dir):
    p = subprocess.run(["semgrep", "scan", "--config", str(ROOT / "rules" / "crypto.yaml"), "--json",
                        "--metrics=off", "--quiet", "--disable-version-check", str(code_dir)],
                       capture_output=True, text=True, encoding="utf-8")
    try:
        res = json.loads(p.stdout)["results"]
    except (json.JSONDecodeError, KeyError):
        print("Semgrep failed:", p.stderr[:500])
        return []
    out = []
    for r in res:
        label = next((lab for lab in LABELS if f"{lab}." in r["check_id"] or r["check_id"].endswith(lab)), None)
        if label:
            out.append({"sample_id": os.path.basename(r["path"])[:-3], "engine": "semgrep",
                        "label": label, "line": r["start"]["line"], "why": r["check_id"].split(".")[-1]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-semgrep", action="store_true")
    ap.add_argument("--no-bandit", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    samples = pd.read_csv(cfg["_out"] / "samples.csv")
    code_dir = out_dir(cfg, "code")
    coded = samples[samples.status == "code"]
    files = [code_dir / f"{s}.py" for s in coded.sample_id]

    findings = []
    for f in files:
        for h in detect(f.read_text(encoding="utf-8")):
            h["why"] = f"{h.pop('engine')}: {h['why']}"  # ast or regex-fallback parser
            findings.append({"sample_id": f.stem, "engine": "custom", **h})
    if not args.no_bandit:
        findings += run_bandit(files)
    if not args.no_semgrep:
        findings += run_semgrep(code_dir)
    F = pd.DataFrame(findings, columns=["sample_id", "engine", "label", "cwe", "severity", "line", "why"])
    F["cwe"] = F.label.map(lambda x: LABELS[x][0])
    F["severity"] = F.label.map(lambda x: LABELS[x][1])
    F.to_csv(cfg["_out"] / "findings.csv", index=False, encoding="utf-8")

    S = samples.copy()
    has_code = S.status == "code"
    for lab in LABELS:
        ids = set(F.loc[F.label == lab, "sample_id"])
        S[lab] = S.sample_id.isin(ids).astype(int).where(has_code)
    for eng in ("custom", "bandit", "semgrep"):
        ids = set(F.loc[(F.engine == eng) & (F.severity == "misuse"), "sample_id"])
        S[f"misuse_{eng}"] = S.sample_id.isin(ids).astype(int).where(has_code)
    S["n_misuse_labels"] = S[MISUSE_LABELS].sum(axis=1).where(has_code)
    S["misuse_any"] = (S["n_misuse_labels"] > 0).astype(int).where(has_code)
    S.to_csv(cfg["_out"] / "scores.csv", index=False, encoding="utf-8")

    print(F.groupby(["engine", "label"]).size().unstack(0, fill_value=0))
    print("\nMisuse rate by language (code samples only):")
    print(S[has_code].groupby(["condition", "lang"]).misuse_any.mean().unstack(0).round(3))


if __name__ == "__main__":
    main()
