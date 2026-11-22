"""Technical validation of the measurement module on synthetic angiographic phantoms.

The public dataset has no ground-truth measurements, so the accuracy of Objective 2
is first established on digital phantoms with KNOWN reference diameter, % diameter
stenosis and lesion length (curved vessels, X-ray-like cylindrical attenuation,
background texture, blur and quantum noise). Reports MAE, bias, RMSE, ICC(2,1) and a
Bland-Altman plot.

  python src/phantom_validation.py --n 300 --out results/phantom
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from measure import measure_stenosis
from stats_utils import icc


def make_phantom(rng, size=256):
    dref = rng.uniform(7, 18)             # px; ~2-4.5 mm at typical 0.25 mm/px
    ds = rng.uniform(0.2, 0.85)           # true fractional diameter stenosis
    sigma = rng.uniform(4, 12)            # lesion half-width parameter (px)
    taper = rng.uniform(-0.15, 0.0)       # mild physiological taper over the segment
    # curved centreline (quadratic Bezier)
    p0 = np.array([rng.uniform(10, 40), rng.uniform(40, size - 40)])
    p2 = np.array([size - rng.uniform(10, 40), rng.uniform(40, size - 40)])
    p1 = (p0 + p2) / 2 + rng.normal(0, 40, 2)
    t = np.linspace(0, 1, 2000)[:, None]
    c = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    s0 = arc[-1] * rng.uniform(0.4, 0.6)
    dref_s = dref * (1 + taper * arc / arc[-1])
    D = dref_s * (1 - ds * np.exp(-((arc - s0) / sigma) ** 2))
    # render: attenuation proportional to chord length through a cylinder
    yy, xx = np.mgrid[0:size, 0:size]
    pts = np.c_[xx.ravel(), yy.ravel()]
    d, i = cKDTree(c).query(pts)
    r = D[i] / 2
    chord = np.sqrt(np.clip(r ** 2 - d ** 2, 0, None)).reshape(size, size)
    bg = 150 + 25 * cv2.GaussianBlur(rng.normal(0, 1, (size, size)), (0, 0), 25) * 8
    img = bg - 70 * chord / (dref / 2)
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # ground truth
    i0 = int(np.argmin(np.abs(arc - s0)))
    ref_true = dref_s[i0]
    mld_true = D[i0]
    ds_true = 100 * (1 - mld_true / ref_true)
    L_true = 2 * sigma * np.sqrt(np.log(ds / 0.2)) if ds > 0.2 else 0.0
    cx, cy = c[i0]
    bw = max(2 * sigma, dref) * 1.2
    box = (cx - bw / 2, cy - bw / 2, bw, bw)
    return img, box, dict(ref_true=ref_true, mld_true=mld_true, ds_true=ds_true, length_true=L_true)


def icc21(a, b):
    return icc(a, b)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="results/phantom")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for k in range(a.n):
        img, box, gt = make_phantom(rng)
        m = measure_stenosis(img, box)
        rows.append({**gt, **m.row()})
        if k < 6:
            from measure import draw
            cv2.imwrite(str(out / f"phantom_{k}.png"), draw(img, box, None, m))
    df = pd.DataFrame(rows)
    df.to_csv(out / "phantom_results.csv", index=False)
    ok = df[df.ok]
    summ = []
    for name, t, p in [("%DS hybrid (primary)", "ds_true", "percent_diameter_stenosis"),
                       ("%DS edge", "ds_true", "percent_ds_edge"),
                       ("%DS densitometric", "ds_true", "percent_ds_densitometric"),
                       ("MLD (px)", "mld_true", "mld_px"),
                       ("Reference D (px)", "ref_true", "ref_diameter_px"),
                       ("Lesion length (px)", "length_true", "lesion_length_px")]:
        e = ok[p] - ok[t]
        summ.append(dict(metric=name, n=len(ok), MAE=e.abs().mean(), bias=e.mean(),
                         RMSE=np.sqrt((e ** 2).mean()), r=np.corrcoef(ok[t], ok[p])[0, 1],
                         ICC21=icc21(ok[t].values, ok[p].values)))
    s = pd.DataFrame(summ)
    s.to_csv(out / "phantom_summary.csv", index=False)
    print(f"Success rate: {df.ok.mean() * 100:.1f}% ({df.ok.sum()}/{len(df)})")
    print(s.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(ok.ds_true, ok.percent_diameter_stenosis, s=10, color="#3b6ea5")
    ax[0].plot([0, 100], [0, 100], "k--", lw=1)
    ax[0].set_xlabel("True %DS"); ax[0].set_ylabel("Measured %DS"); ax[0].set_title("Agreement")
    mean = (ok.ds_true + ok.percent_diameter_stenosis) / 2
    diff = ok.percent_diameter_stenosis - ok.ds_true
    md, sd = diff.mean(), diff.std()
    ax[1].scatter(mean, diff, s=10, color="#3b6ea5")
    for v, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
        ax[1].axhline(v, color="k", ls=ls, lw=1)
    ax[1].set_xlabel("Mean %DS"); ax[1].set_ylabel("Measured - true")
    ax[1].set_title(f"Bland-Altman: bias {md:.1f}, LoA ±{1.96 * sd:.1f}")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "phantom_agreement.png", dpi=150)


if __name__ == "__main__":
    main()
