"""Agreement statistics used for phantom and expert (Dr. Samiullah) validation."""
from __future__ import annotations

import numpy as np
from scipy import stats


def icc(x, y, kind: str = "2,1"):
    """Intraclass correlation (Shrout & Fleiss / McGraw & Wong), 2 raters.

    '2,1' two-way random, absolute agreement, single measurement (recommended)
    '3,1' two-way mixed, consistency, single measurement
    Returns (icc, 95% CI low, 95% CI high).
    """
    Y = np.c_[np.asarray(x, float), np.asarray(y, float)]
    Y = Y[~np.isnan(Y).any(1)]
    n, k = Y.shape
    gm = Y.mean()
    ssr = k * ((Y.mean(1) - gm) ** 2).sum()
    ssc = n * ((Y.mean(0) - gm) ** 2).sum()
    sse = ((Y - gm) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    if kind == "3,1":
        v = (msr - mse) / (msr + (k - 1) * mse)
        f = msr / mse
        fl, fu = f / stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1)), f * stats.f.ppf(0.975, (n - 1) * (k - 1), n - 1)
        return v, (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)
    v = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    # McGraw & Wong CI for ICC(A,1)
    a = k * v / (n * (1 - v))
    b = 1 + k * v * (n - 1) / (n * (1 - v))
    vn = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    fs = stats.f.ppf(0.975, n - 1, vn)
    fi = stats.f.ppf(0.975, vn, n - 1)
    lo = n * (msr - fs * mse) / (fs * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi = n * (fi * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fi * msr)
    return v, lo, hi


def agreement(truth, pred) -> dict:
    t, p = np.asarray(truth, float), np.asarray(pred, float)
    ok = ~(np.isnan(t) | np.isnan(p))
    t, p = t[ok], p[ok]
    e = p - t
    v, lo, hi = icc(t, p)
    tt = stats.ttest_rel(p, t) if len(t) > 1 else None
    w = stats.wilcoxon(p, t) if len(t) > 5 and np.any(e != 0) else None
    return dict(n=len(t), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(),
                LoA_low=e.mean() - 1.96 * e.std(ddof=1), LoA_high=e.mean() + 1.96 * e.std(ddof=1),
                pearson_r=stats.pearsonr(t, p)[0] if len(t) > 2 else np.nan,
                ICC21=v, ICC_CI_low=lo, ICC_CI_high=hi,
                paired_t_p=tt.pvalue if tt else np.nan, wilcoxon_p=w.pvalue if w else np.nan)


def bland_altman(ax, truth, pred, label="", unit=""):
    t, p = np.asarray(truth, float), np.asarray(pred, float)
    m, d = (t + p) / 2, p - t
    md, sd = np.nanmean(d), np.nanstd(d, ddof=1)
    ax.scatter(m, d, s=12, color="#3b6ea5", alpha=0.8)
    for v, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
        ax.axhline(v, color="k", ls=ls, lw=1)
    ax.set_xlabel(f"Mean of methods {unit}")
    ax.set_ylabel(f"Difference {unit}")
    ax.set_title(f"{label} bias {md:.2f}, LoA [{md - 1.96 * sd:.2f}, {md + 1.96 * sd:.2f}]", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
