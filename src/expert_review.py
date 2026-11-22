"""Clinical validation against the expert (Dr. Samiullah) — visual "eyeballing".

The expert has no separate measurement file, so validation is prospective:
the expert reviews AI output on a sample of test cases. To avoid anchoring bias
the review is done in TWO STAGES:

  Stage 1 (blind)  : raw angiogram only  -> stenosis present?  visual %DS, severity
  Stage 2 (AI)     : AI overlay          -> detection correct? measurement acceptable?

Step 1 — build the review pack (images + Excel with drop-downs):
  python src/expert_review.py make --pred results/rtdetr-l/predictions_test.csv --n 200

Step 2 — expert fills  review_pack/expert_review.xlsx  (stage 1 sheet FIRST)

Step 3 — statistics (Cohen's kappa, weighted kappa, ICC, Bland-Altman, paired t-test,
         Wilcoxon, MAE, RMSE) + report:
  python src/expert_review.py analyze --pack results/review_pack
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

SEV = ["mild (<50%)", "moderate (50-69%)", "severe (70-89%)", "critical (>=90%)"]


def make(args, cfg):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    from measure import draw, measure_stenosis

    pred = pd.read_csv(args.pred)
    pack = Path(args.out)
    (pack / "stage1_raw").mkdir(parents=True, exist_ok=True)
    (pack / "stage2_ai").mkdir(parents=True, exist_ok=True)
    img_dir = Path(cfg["data"]["out_dir"]) / "images" / args.split
    # one frame per cine sequence first (diversity), then random fill
    pred["seq"] = pred.image.str.rsplit("_", n=1).str[0]
    rng = np.random.default_rng(args.seed)
    first = pred.groupby("seq", group_keys=False).apply(lambda g: g.sample(1, random_state=args.seed))
    rest = pred.drop(first.index)
    sample = pd.concat([first, rest.sample(max(0, args.n - len(first)), random_state=args.seed)]).head(args.n)
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    key = []
    for i, r in sample.iterrows():
        cid = f"case_{i + 1:03d}"
        img = cv2.imread(str(img_dir / r.image))
        cv2.imwrite(str(pack / "stage1_raw" / f"{cid}.jpg"), img)
        boxes = json.loads(r.pred_boxes)
        ai = dict(case_id=cid, image=r.image, ai_detected=int(len(boxes) > 0))
        if boxes:
            b = max(boxes, key=lambda t: t[4])
            m = measure_stenosis(img, b[:4], cfg["measure"]["mm_per_pixel"])
            vis = draw(img, b[:4], b[4], m)
            ai.update(ai_conf=b[4], ai_ds=m.percent_diameter_stenosis, ai_severity=m.severity,
                      ai_mld_px=m.mld_px, ai_length_px=m.lesion_length_px, ai_ok=m.ok)
        else:
            vis = img.copy()
            cv2.putText(vis, "AI: no stenosis detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        side = np.hstack([img, vis])
        cv2.imwrite(str(pack / "stage2_ai" / f"{cid}.jpg"), side)
        key.append(ai)
    pd.DataFrame(key).to_csv(pack / "ai_key_DO_NOT_SHOW_EXPERT.csv", index=False)

    wb = Workbook()
    hdr_fill = PatternFill("solid", fgColor="DDE6F0")

    def sheet(ws, cols, widths, note):
        ws.append([note])
        ws["A1"].font = Font(bold=True, color="9C2B2B")
        ws.append(cols)
        for c in ws[2]:
            c.font, c.fill = Font(bold=True), hdr_fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for i, w in enumerate(widths):
            ws.column_dimensions[chr(65 + i)].width = w
        ws.freeze_panes = "A3"

    ws1 = wb.active
    ws1.title = "Stage1_Blind"
    sheet(ws1, ["case_id", "image (folder stage1_raw)", "stenosis_present", "visual_DS_percent (0-100)",
                "severity", "vessel", "comments"], [10, 26, 16, 18, 20, 10, 40],
          "STAGE 1 — complete this sheet FIRST using only the images in 'stage1_raw' (no AI output).")
    ws2 = wb.create_sheet("Stage2_AI_Review")
    sheet(ws2, ["case_id", "image (folder stage2_ai)", "detection", "box_location_correct",
                "measurement_acceptable", "comments"], [10, 26, 22, 18, 22, 40],
          "STAGE 2 — only after Stage 1 is finished. Left = raw, right = AI output.")
    n = len(sample)
    for i in range(n):
        cid = f"case_{i + 1:03d}"
        ws1.append([cid, f"stage1_raw/{cid}.jpg"])
        ws2.append([cid, f"stage2_ai/{cid}.jpg"])
    last = n + 2

    def dv(ws, formula, rng_):
        v = DataValidation(type="list", formula1=formula, allow_blank=True)
        ws.add_data_validation(v)
        v.add(rng_)

    dv(ws1, '"Yes,No"', f"C3:C{last}")
    v = DataValidation(type="decimal", operator="between", formula1="0", formula2="100", allow_blank=True)
    ws1.add_data_validation(v)
    v.add(f"D3:D{last}")
    dv(ws1, '"' + ",".join(SEV) + '"', f"E3:E{last}")
    dv(ws1, '"LAD,LCX,RCA,LM,Other"', f"F3:F{last}")
    dv(ws2, '"Correct,Partially correct,Wrong (false positive),Missed (false negative),Correct - no lesion"', f"C3:C{last}")
    dv(ws2, '"Yes,No"', f"D3:D{last}")
    dv(ws2, '"Yes,No"', f"E3:E{last}")
    wb.save(pack / "expert_review.xlsx")
    (pack / "INSTRUCTIONS.txt").write_text(
        "Expert review instructions\n==========================\n"
        "1. Open expert_review.xlsx.\n"
        "2. Sheet 'Stage1_Blind': open each image in stage1_raw/ and record, by visual assessment only,\n"
        "   whether a significant stenosis is present, the estimated % diameter stenosis, severity and vessel.\n"
        "   Do NOT look at stage2_ai/ before finishing Stage 1.\n"
        "3. Sheet 'Stage2_AI_Review': open the matching image in stage2_ai/ (left raw, right AI) and judge\n"
        "   whether the AI detection and its measurement are acceptable.\n"
        "4. Return expert_review.xlsx. Do not open ai_key_DO_NOT_SHOW_EXPERT.csv.\n")
    print(f"Review pack with {n} cases written to {pack.resolve()}")


def analyze(args, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    from stats_utils import agreement, bland_altman

    pack = Path(args.pack)
    key = pd.read_csv(pack / "ai_key_DO_NOT_SHOW_EXPERT.csv")
    s1 = pd.read_excel(pack / "expert_review.xlsx", sheet_name="Stage1_Blind", header=1)
    s2 = pd.read_excel(pack / "expert_review.xlsx", sheet_name="Stage2_AI_Review", header=1)
    s1.columns = ["case_id", "image", "present", "visual_ds", "severity", "vessel", "comments"]
    s2.columns = ["case_id", "image", "detection", "loc_ok", "meas_ok", "comments"]
    df = key.merge(s1, on="case_id").merge(s2[["case_id", "detection", "loc_ok", "meas_ok"]], on="case_id")
    df = df[df.present.notna()]
    out = pack / "analysis"
    out.mkdir(exist_ok=True)
    rep = ["# Expert validation report", "", f"Cases reviewed: **{len(df)}**", ""]

    # --- detection agreement (AI detected vs expert says present)
    e = (df.present == "Yes").astype(int)
    a = df.ai_detected.astype(int)
    tn, fp, fn, tp = confusion_matrix(e, a, labels=[0, 1]).ravel()
    kappa = cohen_kappa_score(e, a) if e.nunique() > 1 or a.nunique() > 1 else np.nan
    rep += ["## 1. Detection (AI vs blind expert)", "",
            "| | Expert: present | Expert: absent |", "|---|---|---|",
            f"| AI: detected | {tp} | {fp} |", f"| AI: not detected | {fn} | {tn} |", "",
            f"- Agreement: {(tp + tn) / len(df) * 100:.1f}%",
            f"- Cohen's kappa: {kappa:.3f}",
            f"- Sensitivity: {tp / max(tp + fn, 1) * 100:.1f}%   Specificity: "
            + (f"{tn / (tn + fp) * 100:.1f}%" if (tn + fp) else "n/a (no expert-negative cases)"),
            f"- PPV: {tp / max(tp + fp, 1) * 100:.1f}%", ""]

    # --- stage 2
    if df.detection.notna().any():
        vc = df.detection.value_counts(normalize=True) * 100
        rep += ["## 2. Expert judgement of AI output (stage 2)", ""] + \
               [f"- {k}: {v:.1f}%" for k, v in vc.items()] + \
               [f"- Box location correct: {(df.loc_ok == 'Yes').mean() * 100:.1f}%",
                f"- Measurement acceptable: {(df.meas_ok == 'Yes').mean() * 100:.1f}%", ""]

    # --- %DS agreement
    mm = df[df.visual_ds.notna() & df.ai_ds.notna()]
    if len(mm) > 2:
        ag = agreement(mm.visual_ds, mm.ai_ds)
        rep += ["## 3. Stenosis severity: AI %DS vs expert visual %DS", "",
                "| Statistic | Value |", "|---|---|"] + \
               [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in ag.items()] + \
               ["", "Note: visual estimation is known to over-estimate stenosis relative to QCA, "
                    "so a negative bias (AI minus expert) is expected.", ""]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].scatter(mm.visual_ds, mm.ai_ds, s=14, color="#3b6ea5")
        ax[0].plot([0, 100], [0, 100], "k--", lw=1)
        ax[0].set_xlabel("Expert visual %DS"); ax[0].set_ylabel("AI %DS")
        ax[0].spines[["top", "right"]].set_visible(False)
        bland_altman(ax[1], mm.visual_ds, mm.ai_ds, "AI - expert:", "(%DS)")
        fig.tight_layout(); fig.savefig(out / "ds_agreement.png", dpi=150)
        pd.Series(ag).to_csv(out / "ds_agreement.csv", header=["value"])

    # --- severity category agreement
    sv = df[df.severity.isin(SEV) & df.ai_severity.isin(SEV)]
    if len(sv) > 2:
        x = sv.severity.map(SEV.index)
        y = sv.ai_severity.map(SEV.index)
        wk = cohen_kappa_score(x, y, weights="quadratic")
        ct = pd.crosstab(sv.severity, sv.ai_severity).reindex(index=SEV, columns=SEV, fill_value=0)
        ct.to_csv(out / "severity_crosstab.csv")
        rep += ["## 4. Severity category (quadratic-weighted kappa)", "",
                f"- Weighted kappa: {wk:.3f}  (n={len(sv)})",
                f"- Exact category agreement: {(x == y).mean() * 100:.1f}%", "",
                "Rows = expert, columns = AI.", "", ct.to_markdown(), ""]
    (out / "expert_validation_report.md").write_text("\n".join(rep))
    df.to_csv(out / "merged_review.csv", index=False)
    print("\n".join(rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "analyze"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None, help="prepared dataset folder (default: data.out_dir in config)")
    ap.add_argument("--pred", help="predictions_<split>.csv from evaluate.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/review_pack")
    ap.add_argument("--pack", default="results/review_pack")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.data_dir:
        cfg["data"]["out_dir"] = args.data_dir
    make(args, cfg) if args.cmd == "make" else analyze(args, cfg)


if __name__ == "__main__":
    main()
