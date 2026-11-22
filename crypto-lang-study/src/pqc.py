"""Step 7: post-quantum readiness score for every answer (RQ3).

Works on the full answer text (any language) plus the extracted code.
    pqc_score 0 = no mention of quantum risk or PQC
              1 = mentions the quantum threat only
              2 = recommends a PQC algorithm in the text
              3 = uses a PQC algorithm in the code (e.g. liboqs / pqcrypto)
Also: classical_pk_in_code (RSA/ECC used) and hybrid (PQC + classical in code).

Output: results/pqc.csv
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import load_config, out_dir, read_jsonl  # noqa: E402
from extract import extract_code  # noqa: E402

PQC_TEXT = re.compile(
    r"ML[-_ ]?KEM|Kyber|CRYSTALS|ML[-_ ]?DSA|Dilithium|SLH[-_ ]?DSA|SPHINCS|Falcon|FN[-_ ]?DSA|HQC|"
    r"FIPS[ -]?20[345]|post[- ]?quantum|quantum[- ](?:safe|resistant|secure)|\bPQC\b|"
    r"后量子|抗量子|ما بعد الكم|مقاوم(?:ة)? للكم|پوسٹ[- ]?کوانٹم|کوانٹم[- ]?(?:محفوظ|مزاحم)|post[- ]?quantam",
    re.I)
QUANTUM_TEXT = re.compile(r"quantum|quantam|量子|الكم|الكمومي|کوانٹم|کوانٹم", re.I)
PQC_CODE = re.compile(r"\bimport\s+oqs\b|from\s+oqs\b|oqs\.(KeyEncapsulation|Signature)|pqcrypto|kyber|mlkem|ml_kem|"
                      r"dilithium|ml_dsa|mldsa|sphincs|slh_dsa|falcon", re.I)


def _strip_comments(code: str) -> str:
    code = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', " ", code)  # docstrings
    return re.sub(r"#[^\n]*", " ", code)
CLASSICAL_PK_CODE = re.compile(r"\brsa\b|RSA\.|ec\.generate_private_key|ECDH|ECDSA|X25519|Ed25519|X448|Ed448|"
                               r"ecdsa|nacl\.public|SECP|dh\.generate_parameters", re.I)


def main():
    cfg = load_config()
    rows = {}
    for f in sorted(out_dir(cfg, "raw").glob("*.jsonl")):
        for r in read_jsonl(f):
            if r.get("response") is not None:
                rows[r["sample_id"]] = r
    out = []
    for sid, r in rows.items():
        text = r["response"]
        code = extract_code(text)
        prose = re.sub(r"```.*?```", " ", text, flags=re.S)
        live = _strip_comments(code)
        uses = bool(PQC_CODE.search(live))
        recommends = bool(PQC_TEXT.search(prose)) or bool(PQC_TEXT.search(code))
        threat = bool(QUANTUM_TEXT.search(text))
        score = 3 if uses else 2 if recommends else 1 if threat else 0
        classical = bool(CLASSICAL_PK_CODE.search(live))
        out.append({"sample_id": sid, "model": r["model"], "condition": r["condition"], "task": r["task"],
                    "lang": r["lang"], "long_term": r["long_term"], "pqc_score": score,
                    "pqc_any": int(score >= 2), "quantum_threat_mentioned": int(threat),
                    "pqc_in_code": int(uses), "classical_pk_in_code": int(classical),
                    "hybrid": int(uses and classical)})
    df = pd.DataFrame(out)
    df.to_csv(cfg["_out"] / "pqc.csv", index=False, encoding="utf-8")
    lt = df[df.long_term]
    print("PQC recommended or used (score >= 2), long-term tasks only:")
    print(lt.groupby(["condition", "lang"]).pqc_any.mean().unstack(0).round(3))


if __name__ == "__main__":
    main()
