"""Clinical validation of the measurement module (Objective 2) against CADICA lesion grades.

CADICA cardiologists graded every annotated lesion into stenosis classes
(<20 %, 20-49 %, 50-69 %, 70-89 %, 90-98 %, 99 %, 100 %). This script runs the automatic
QCA-style measurement (measure.py) inside every graded box - on the RAW CADICA frames,
independent of the detector - and reports how well the automatic % diameter stenosis
agrees with the cardiologists' visual grade.

Unit of analysis
  primary   : lesion = one graded box track per video (median %DS over its key frames)
  secondary : every annotated frame

Outputs  (results/measurement_cadica/)
  frames.csv, lesions.csv            raw measurements
  summary.csv                        n, success rate, Spearman rho, weighted kappa,
                                     AUC (>=50 %, >=70 %) with patient-cluster bootstrap CI,
                                     Youden threshold, sensitivity/specificity at that threshold
  confusion_5class.csv               AI class vs cardiologist class (lesion level)
  ds_by_grade.png                    box plot of automatic %DS per cardiologist grade

  python src/validate_measurement_cadica.py
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve

from measure import measure_stenosis
from prepare_data import find_dir
from preprocess import enhance

GRADES = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]
LOWER = {"p0_20": 0, "p20_50": 20, "p50_70": 50, "p70_90": 70, "p90_98": 90, "p99": 99, "p100": 100}
MID = {"p0_20": 10, "p20_50": 35, "p50_70": 60, "p70_90": 80, "p90_98": 94, "p99": 99, "p100": 100}
CLASSES = ["<20%", "20-49%", "50-69%", "70-89%", ">=90%"]          # 99 % and 100 % merged into >=90 %


def grade_class(g: str) -> int:
    lo = LOWER[g]
    return 0 if lo < 20 else 1 if lo < 50 else 2 if lo < 70 else 3 if lo < 90 else 4


def ds_class(ds: float) -> int:
    return 0 if ds < 20 else 1 if ds < 50 else 2 if ds < 70 else 3 if ds < 90 else 4


def norm_grade(label: str) -> str | None:
    m = re.match(r"p(\d+)(?:_(\d+))?", label.strip().lower())
    if not m:
        return None
    g = f"p{m.group(1)}" + (f"_{m.group(2)}" if m.group(2) else "")
    return g if g in LOWER else None


def collect(root: Path, cfg, limit: int = 0):
    pp, mc = cfg["preprocess"], cfg["measure"]
    rows = []
    sel = root / "selectedVideos"
    for pdir in sorted(sel.glob("p*"), key=lambda p: int(p.name[1:])):
        lv = (pdir / "lesionVideos.txt").read_text().split() if (pdir / "lesionVideos.txt").exists() else []
        for v in lv:
            vdir = pdir / v
            for gt in sorted((vdir / "groundtruth").glob(f"{pdir.name}_{v}_*.txt")):
                img_p = vdir / "input" / f"{gt.stem}.png"
                if not img_p.exists():
                    continue
                img = None
                for k, line in enumerate(gt.read_text().strip().splitlines()):
                    parts = line.split()
                    if len(parts) < 5 or norm_grade(parts[4]) is None:
                        continue
                    if img is None:
                        raw = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
                        img = enhance(raw, pp["clahe"], pp["clahe_clip"], pp["clahe_tile"], pp["denoise"])
                    box = [float(x) for x in parts[:4]]
                    t = time.perf_counter()
                    m = measure_stenosis(img, box, mc["mm_per_pixel"], mc["box_margin"], mc["min_crop"],
                                         mc["lesion_threshold"])
                    g = norm_grade(parts[4])
                    rows.append(dict(patient=pdir.name, video=f"{pdir.name}_{v}", frame=gt.stem, box_idx=k,
                                     x=box[0], y=box[1], w=box[2], h=box[3], grade=g, grade_lower=LOWER[g],
                                     grade_mid=MID[g], ms=(time.perf_counter() - t) * 1000, **m.row()))
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def lesion_level(fr: pd.DataFrame) -> pd.DataFrame:
    """One row per lesion: boxes of the same grade in a video, linked by nearest centre."""
    fr = fr.assign(cx=fr.x + fr.w / 2, cy=fr.y + fr.h / 2)
    out = []
    for (vid, g), grp in fr.groupby(["video", "grade"]):
        tracks = []                                     # list of [cx, cy, rows]
        for _, r in grp.sort_values("frame").iterrows():
            d = [np.hypot(r.cx - t[0], r.cy - t[1]) for t in tracks]
            if d and min(d) < 60:
                t = tracks[int(np.argmin(d))]
                t[0], t[1] = r.cx, r.cy
                t[2].append(r)
            else:
                tracks.append([r.cx, r.cy, [r]])
        for i, t in enumerate(tracks):
            tdf = pd.DataFrame(t[2])
            ok = tdf[tdf.ok]
            out.append(dict(patient=tdf.patient.iloc[0], video=vid, lesion=f"{vid}_{g}_{i}", grade=g,
                            grade_lower=LOWER[g], grade_mid=MID[g], frames=len(tdf), measured=len(ok),
                            ds=float(ok.percent_diameter_stenosis.median()) if len(ok) else np.nan,
                            ds_edge=float(ok.percent_ds_edge.median()) if len(ok) else np.nan,
                            ds_dens=float(ok.percent_ds_densitometric.median()) if len(ok) else np.nan,
                            length_px=float(ok.lesion_length_px.median()) if len(ok) else np.nan,
                            mld_px=float(ok.mld_px.median()) if len(ok) else np.nan))
    return pd.DataFrame(out)


def cluster_boot_auc(df, y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pats = df.patient.unique()
    vals = []
    for _ in range(n):
        pick = rng.choice(pats, len(pats), replace=True)
        b = pd.concat([df[df.patient == p] for p in pick])
        if b[y].nunique() == 2:
            vals.append(roc_auc_score(b[y], b[s]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def analyse(les: pd.DataFrame, fr: pd.DataFrame, out: Path):
    L = les.dropna(subset=["ds"]).copy()
    res = dict(frames_annotated=len(fr), frames_measured=int(fr.ok.sum()),
               frame_success_rate=float(fr.ok.mean()), lesions=len(les), lesions_measured=len(L),
               patients=int(L.patient.nunique()), measure_ms_median=float(fr.ms.median()))
    rho, p = stats.spearmanr(L.ds, L.grade_mid)
    res.update(spearman_rho=rho, spearman_p=p)
    yt = L.grade.map(grade_class)
    yp = L.ds.map(ds_class)
    res.update(weighted_kappa_quadratic=cohen_kappa_score(yt, yp, weights="quadratic"),
               exact_class_agreement=float((yt == yp).mean()),
               within_one_class=float((np.abs(yt - yp) <= 1).mean()))
    cm = pd.DataFrame(confusion_matrix(yt, yp, labels=range(5)), index=[f"cardiologist {c}" for c in CLASSES],
                      columns=[f"AI {c}" for c in CLASSES])
    cm.to_csv(out / "confusion_5class.csv")
    for thr in (50, 70):
        col = f"ge{thr}"
        L[col] = (L.grade_lower >= thr).astype(int)
        if L[col].nunique() == 2:
            auc = roc_auc_score(L[col], L.ds)
            lo, hi = cluster_boot_auc(L, col, "ds")
            fpr, tpr, th = roc_curve(L[col], L.ds)
            j = int(np.argmax(tpr - fpr))
            res.update({f"AUC_ge{thr}": auc, f"AUC_ge{thr}_CI_low": lo, f"AUC_ge{thr}_CI_high": hi,
                        f"youden_threshold_ge{thr}": float(th[j]), f"sens_at_youden_ge{thr}": float(tpr[j]),
                        f"spec_at_youden_ge{thr}": float(1 - fpr[j]),
                        f"sens_at_DS{thr}_ge{thr}": float(((L.ds >= thr) & (L[col] == 1)).sum() / max(L[col].sum(), 1)),
                        f"spec_at_DS{thr}_ge{thr}": float(((L.ds < thr) & (L[col] == 0)).sum() / max((L[col] == 0).sum(), 1))})
    by = L.groupby("grade").ds.describe()[["count", "mean", "50%", "25%", "75%"]].reindex(
        [g for g in GRADES if g in set(L.grade)])
    by.to_csv(out / "ds_by_grade.csv")
    pd.Series(res).to_csv(out / "summary.csv", header=["value"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gs = [g for g in GRADES if g in set(L.grade)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = [L[L.grade == g].ds.values for g in gs]
    ax.boxplot(data, showfliers=False, widths=0.55, medianprops=dict(color="#0d6b6b", lw=2))
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.07, len(d)), d, s=8, alpha=0.45, color="#3b6ea5")
    for i, g in enumerate(gs, 1):                                   # cardiologist grade range as grey band
        lo, hi = LOWER[g], {"p0_20": 20, "p20_50": 50, "p50_70": 70, "p70_90": 90, "p90_98": 98, "p99": 99, "p100": 100}[g]
        ax.fill_between([i - 0.3, i + 0.3], lo, max(hi, lo + 1), color="grey", alpha=0.15, lw=0)
    ax.set_xticks(range(1, len(gs) + 1), [g.replace("p", "").replace("_", "-") + "%" for g in gs])
    ax.set_xlabel("Cardiologist visual grade (CADICA)")
    ax.set_ylabel("Automatic % diameter stenosis")
    ax.set_title(f"Lesion level, n = {len(L)}; Spearman rho = {rho:.2f}; grey = grade range", fontsize=10)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "ds_by_grade.png", dpi=170)
    return res, cm, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cadica", default=None, help="CADICA folder (default: data.cadica_dir in config)")
    ap.add_argument("--out", default="results/measurement_cadica")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N boxes")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    root = find_dir(a.cadica or cfg["data"]["cadica_dir"], "metadata.xlsx")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Measuring graded CADICA lesions in {root} ...")
    fr = collect(root, cfg, a.limit)
    fr.to_csv(out / "frames.csv", index=False)
    les = lesion_level(fr)
    les.to_csv(out / "lesions.csv", index=False)
    res, cm, by = analyse(les, fr, out)
    pd.set_option("display.width", 160)
    for k, v in res.items():
        print(f"{k:32s} {v:.4f}" if isinstance(v, float) else f"{k:32s} {v}")
    print("\nAutomatic %DS by cardiologist grade (lesion level):")
    print(by.round(1).to_string())
    print("\nConfusion (rows = cardiologist, columns = AI):")
    print(cm.to_string())
    print(f"\nResults in {out.resolve()}")


if __name__ == "__main__":
    main()
