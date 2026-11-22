"""Step 6: manual review of a random 20% sample, and agreement statistics (Table 5).

1) python src/manual.py sample      -> results/manual/rater_A.csv, rater_B.csv
   Each rater fills the label columns with 1 (present) or 0 (absent), independently.
   The sheet does not show the prompt language or the model (blind review); the task is shown
   in English. Open the code file named in `code_file`.
2) python src/manual.py agree       -> Cohen's kappa per label between raters,
                                       results/manual/disagreements.csv to adjudicate
3) Fill the `final_*` columns in disagreements.csv together, then
   python src/manual.py final       -> results/manual/consensus.csv and tool-vs-manual table
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import load_config, load_prompts, out_dir  # noqa: E402
from detectors import LABELS  # noqa: E402

MISUSE = [k for k, v in LABELS.items() if v[1] == "misuse"]
FRACTION = 0.20
SEED = 42


def cohen_kappa(a, b):
    a, b = pd.Series(a).astype(int).reset_index(drop=True), pd.Series(b).astype(int).reset_index(drop=True)
    n = len(a)
    if n == 0:
        return float("nan")
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in (0, 1))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cmd_sample(cfg):
    S = pd.read_csv(cfg["_out"] / "scores.csv")
    S = S[S.status == "code"]
    P = {t["id"]: t["en"] for t in load_prompts(cfg)["tasks"]}
    pick = (S.groupby(["model", "condition", "lang"], group_keys=False)
             .apply(lambda g: g.sample(frac=FRACTION, random_state=SEED)))
    pick = pick.sample(frac=1, random_state=SEED)  # shuffle so languages are mixed
    sheet = pd.DataFrame({
        "review_id": [f"R{i:04d}" for i in range(len(pick))],
        "sample_id": pick.sample_id.values,
        "code_file": [f"results/code/{s}.py" for s in pick.sample_id],
        "task_en": pick.task.map(P).values,
    })
    for lab in MISUSE:
        sheet[lab] = ""
    sheet["other_problem"] = ""
    sheet["notes"] = ""
    d = out_dir(cfg, "manual")
    for r in ("A", "B"):
        sheet.drop(columns=["sample_id"]).to_csv(d / f"rater_{r}.csv", index=False, encoding="utf-8-sig")
    sheet[["review_id", "sample_id"]].to_csv(d / "key.csv", index=False, encoding="utf-8")
    rubric = pd.DataFrame([{"label": k, "cwe": v[0], "definition": v[2]} for k, v in LABELS.items() if v[1] == "misuse"])
    rubric.to_csv(d / "rubric.csv", index=False, encoding="utf-8-sig")
    print(f"{len(sheet)} samples written to {d} (rater_A.csv, rater_B.csv, rubric.csv)")


def _load_raters(d):
    A = pd.read_csv(d / "rater_A.csv", encoding="utf-8-sig")
    B = pd.read_csv(d / "rater_B.csv", encoding="utf-8-sig")
    for X in (A, B):
        X[MISUSE] = X[MISUSE].fillna(0).astype(int)
        X["misuse_any"] = (X[MISUSE].sum(axis=1) > 0).astype(int)
    return A.set_index("review_id"), B.set_index("review_id")


def cmd_agree(cfg):
    d = out_dir(cfg, "manual")
    A, B = _load_raters(d)
    rows = []
    for lab in MISUSE + ["misuse_any"]:
        rows.append({"label": lab, "n": len(A), "agreement_%": round(100 * (A[lab] == B[lab]).mean(), 1),
                     "kappa": round(cohen_kappa(A[lab], B[lab]), 3),
                     "prevalence_A_%": round(100 * A[lab].mean(), 1)})
    T = pd.DataFrame(rows)
    T.to_csv(d / "table5a_inter_rater.csv", index=False)
    print(T.to_string(index=False))
    dis = A.index[(A[MISUSE] != B[MISUSE]).any(axis=1)]
    out = A.loc[dis, ["code_file", "task_en"]].copy()
    for lab in MISUSE:
        out[f"A_{lab}"], out[f"B_{lab}"], out[f"final_{lab}"] = A.loc[dis, lab], B.loc[dis, lab], ""
    out.to_csv(d / "disagreements.csv", encoding="utf-8-sig")
    print(f"\n{len(dis)} disagreements -> {d / 'disagreements.csv'}")


def cmd_final(cfg):
    d = out_dir(cfg, "manual")
    A, B = _load_raters(d)
    C = A[MISUSE].copy()
    dis_path = d / "disagreements.csv"
    if dis_path.exists():
        D = pd.read_csv(dis_path, encoding="utf-8-sig").set_index("review_id")
        for lab in MISUSE:
            col = D[f"final_{lab}"]
            if col.isna().any():
                raise SystemExit(f"Fill final_{lab} for every row in disagreements.csv first")
            C.loc[D.index, lab] = col.astype(int)
    C["misuse_any"] = (C[MISUSE].sum(axis=1) > 0).astype(int)
    key = pd.read_csv(d / "key.csv").set_index("review_id")
    C = C.join(key)
    C.to_csv(d / "consensus.csv", encoding="utf-8")

    S = pd.read_csv(cfg["_out"] / "scores.csv").set_index("sample_id")
    rows = []
    for col_tool, name in [("misuse_any", "all engines (union)"), ("misuse_custom", "custom AST"),
                           ("misuse_bandit", "Bandit"), ("misuse_semgrep", "Semgrep")]:
        t = S.loc[C.sample_id, col_tool].values.astype(int)
        m = C.misuse_any.values
        tp, fp, fn = int(((t == 1) & (m == 1)).sum()), int(((t == 1) & (m == 0)).sum()), int(((t == 0) & (m == 1)).sum())
        rows.append({"engine": name, "kappa_vs_manual": round(cohen_kappa(t, m), 3),
                     "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                     "recall": round(tp / (tp + fn), 3) if tp + fn else None, "n": len(m)})
    T = pd.DataFrame(rows)
    T.to_csv(d / "table5b_tool_vs_manual.csv", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    cfg = load_config()
    {"sample": cmd_sample, "agree": cmd_agree, "final": cmd_final}[sys.argv[1] if len(sys.argv) > 1 else "sample"](cfg)
