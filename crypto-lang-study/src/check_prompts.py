"""Part C, step 1: automatic checks on data/prompts.json before data collection.

Static checks (no model needed, always run):
  1. every task has all six languages, none empty
  2. no security words in the prompts (except the two long-term tasks, where the time
     requirement is the point) - so the model is not nudged towards secure code
  3. right script: ur/ar in Arabic script, zh in Chinese characters, ru/cm in Latin letters
  4. numbers and acronyms in the English prompt (6, 20, JWT, HMAC, RSA ...) are kept in each version

Back-translation (optional, needs Ollama):  --backtranslate llama3.1:8b-instruct-q4_K_M
  Each non-English prompt is translated back into English by a local model and compared with the
  English source using chrF (character n-gram F-score, 0-100, as in MT evaluation). Low scores are
  flagged for a human to look at. A machine back-translation does NOT replace the native-speaker
  check reported in the paper: it only tells the human which rows to look at first.

Output: results/prompt_check/prompt_check.csv  (open in Excel)  +  a summary on screen.
Exit code 1 if a hard problem (missing text, wrong script, security word) is found.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from common import load_config, load_prompts, out_dir  # noqa: E402

LANG_NAME = {"en": "English", "ur": "Urdu", "ru": "Roman Urdu", "cm": "Code-mixed", "zh": "Chinese", "ar": "Arabic"}

SECURITY_WORDS = {
    "en": r"\bsecure\w*|\bsecurity\b|\bsafe(ly|ty)?\b|\bprotect\w*|\bvulnerab\w*|best practice",
    "ru": r"\bsecure\w*|\bsecurity\b|\bsafe\b|\bma?h?fooz\b|\bmehfooz\b|\bhifazat\b|\bsurakshit\b|\bbest practice",
    "cm": r"\bsecure\w*|\bsecurity\b|\bsafe\b|\bma?h?fooz\b|\bmehfooz\b|\bhifazat\b|\bsurakshit\b|\bbest practice",
    "ur": r"محفوظ|حفاظت|سیکیور|سکیور|سیکورٹی|سیکیورٹی",
    "zh": r"安全|保护|防护",
    "ar": r"آمن|أمان|أمن|حماية|محمي",
}
SCRIPT = {
    "ur": (r"[؀-ۿ]", 0.6), "ar": (r"[؀-ۿ]", 0.6),
    "zh": (r"[一-鿿]", 0.4), "ru": (r"[A-Za-z]", 0.9), "cm": (r"[A-Za-z]", 0.9),
}
TOKEN = re.compile(r"\b(?:[A-Z][A-Z0-9]{1,}|\d+)\b")  # acronyms and numbers
# acceptable local renderings of some English tokens
ALIASES = {
    "6": ["چھ", "six", "chhe", "chay", "六", "ستة", "6"],
    "SMS": ["短信", "ایس ایم ایس", "SMS"],
    "API": ["API", "اے پی آئی"],
    "JSON": ["JSON"],
}


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popovic 2015): character n-gram F-beta, spaces removed, 0-100."""
    h, r = re.sub(r"\s+", "", hyp.lower()), re.sub(r"\s+", "", ref.lower())
    precs, recs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        overlap = sum((hc & rc).values())
        precs.append(overlap / sum(hc.values()))
        recs.append(overlap / sum(rc.values()))
    if not precs:
        return 0.0
    p, rr = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + rr == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


def back_translate(cfg, model: str, text: str, lang: str) -> str:
    msg = ("Translate the following developer request into English. Output only the English "
           f"translation, nothing else. The text is in {LANG_NAME[lang]}"
           + (" (Urdu written in Latin letters)" if lang == "ru" else "")
           + (" (a mix of Urdu/Hindi and English)" if lang == "cm" else "") + f":\n\n{text}")
    r = requests.post(f"{cfg['ollama_url']}/api/chat", timeout=600, json={
        "model": model, "messages": [{"role": "user", "content": msg}], "stream": False,
        "options": {"temperature": 0, "seed": 1, "num_predict": 300}})
    r.raise_for_status()
    return r.json()["message"]["content"].strip().strip('"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtranslate", metavar="OLLAMA_MODEL",
                    help="local model used for back-translation, e.g. llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--threshold", type=float, default=45.0, help="chrF below this is flagged (default 45)")
    args = ap.parse_args()
    cfg = load_config()
    P = load_prompts(cfg)
    langs = P["languages"]
    rows, hard = [], 0

    for t in P["tasks"]:
        en = t["en"]
        en_tokens = set(TOKEN.findall(en))
        for lang in langs:
            text = (t.get(lang) or "").strip()
            issues = []
            if not text:
                issues.append("EMPTY")
            rx = SECURITY_WORDS.get(lang)
            if text and rx and not t.get("long_term"):
                hits = sorted({m.group(0) for m in re.finditer(rx, text, flags=re.I)})
                if hits:
                    issues.append("SECURITY WORD: " + ", ".join(hits))
            if text and lang in SCRIPT:
                pat, need = SCRIPT[lang]
                letters = [c for c in text if c.isalpha()]
                share = sum(bool(re.match(pat, c)) for c in letters) / max(1, len(letters))
                if share < need:
                    issues.append(f"WRONG SCRIPT ({share:.0%} expected script)")
            warns = []
            if text and lang != "en":
                for tok in sorted(en_tokens):
                    options = ALIASES.get(tok, [tok])
                    if not any(o.lower() in text.lower() for o in options):
                        warns.append(tok)
            if issues:
                hard += 1
            rows.append({"task": t["id"], "lang": lang, "language": LANG_NAME[lang], "prompt": text,
                         "problems": "; ".join(issues), "missing_terms": ", ".join(warns),
                         "english_source": en})

    df = pd.DataFrame(rows)
    df["identical_to_english"] = (df.lang != "en") & (df.prompt.str.strip() == df.english_source.str.strip())
    if df.identical_to_english.any():
        hard += int(df.identical_to_english.sum())
        df.loc[df.identical_to_english, "problems"] += "; NOT TRANSLATED"

    if args.backtranslate:
        print(f"Back-translating {int((df.lang != 'en').sum())} prompts with {args.backtranslate} ...")
        bts, scores = [], []
        for i, r in df.iterrows():
            if r.lang == "en" or not r.prompt:
                bts.append("")
                scores.append(None)
                continue
            try:
                bt = back_translate(cfg, args.backtranslate, r.prompt, r.lang)
            except Exception as e:  # noqa: BLE001
                bt = f"ERROR: {e}"
            bts.append(bt)
            scores.append(round(chrf(bt, r.english_source), 1) if not bt.startswith("ERROR") else None)
            print(f"  {r.task} {r.lang}: chrF={scores[-1]}")
        df["back_translation"] = bts
        df["chrF"] = scores
        df["needs_human_look"] = (df.chrF.notna() & (df.chrF < args.threshold)) | (df.missing_terms != "") | (df.problems != "")
        df["back_translation_model"] = args.backtranslate
    else:
        df["needs_human_look"] = (df.missing_terms != "") | (df.problems != "")

    df["human_ok"] = ""        # native speaker writes yes / no
    df["human_comment"] = ""
    d = out_dir(cfg, "prompt_check")
    df.to_csv(d / "prompt_check.csv", index=False, encoding="utf-8-sig")

    print("\n=== Prompt check summary ===")
    print(f"Tasks: {len(P['tasks'])}, languages: {len(langs)}, prompts: {len(df)}")
    probs = df[df.problems != ""]
    print(f"Hard problems: {len(probs)}")
    for _, r in probs.iterrows():
        print(f"  {r.task} {r.language}: {r.problems}")
    miss = df[df.missing_terms != ""]
    print(f"Warnings (a number/acronym from English not found - often fine, e.g. written as a word): {len(miss)}")
    for _, r in miss.iterrows():
        print(f"  {r.task} {r.language}: {r.missing_terms}")
    if "chrF" in df:
        s = df[df.chrF.notna()].groupby("language").chrF.agg(["mean", "min"]).round(1)
        print("\nBack-translation chrF by language (higher = closer to English source):")
        print(s.to_string())
    print(f"\nRows to look at first: {int(df.needs_human_look.sum())} of {len(df)}")
    print(f"Report: {d / 'prompt_check.csv'}  (filter column needs_human_look = TRUE)")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
