"""Step 9: Figures 1-5 (PNG at 300 dpi + PDF) into results/figures/.

Run after stats.py. Colours: fixed categorical order, one colour per model, validated for
colour-vision deficiency; every figure also uses markers / labels so colour is never the only cue.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import load_config, out_dir  # noqa: E402
from stats import LANG_NAME, LANG_ORDER  # noqa: E402
from detectors import LABELS  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, d, name):
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1(d):
    steps = ["20 crypto\ntasks", "6 prompt\nlanguages\n(+ back-translation)", "LLMs\n(open + commercial)\n× 5 samples",
             "Code extraction\n(refusals and\nno-code logged)", "Scoring\nAST + Bandit +\nSemgrep + 20% manual", "Statistics\nRQ1–RQ4"]
    fig, ax = plt.subplots(figsize=(11, 1.7))
    ax.set_axis_off()
    w, gap = 1.6, 0.3
    for i, s in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#f0efec", ec=AXIS, lw=1))
        ax.text(x + w / 2, 0.5, s, ha="center", va="center", fontsize=7.5, color=INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 0.5), xytext=(x + w + 0.02, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(2 * (w + gap) + w / 2, -0.22, "baseline and secure system prompt (RQ4)", ha="center",
            fontsize=7.5, color=INK2, style="italic")
    ax.set_xlim(-0.05, len(steps) * (w + gap)); ax.set_ylim(-0.35, 1.05)
    save(fig, d, "fig1_pipeline")


def _langs(df):
    return [lang for lang in LANG_ORDER if lang in set(df.lang)]


def fig_rate_by_model(T, d, name, ylabel, title_all="All models"):
    langs = _langs(T)
    models = sorted(T.model.unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(langs))
    k = len(models)
    offs = np.linspace(-0.28, 0.28, k) if k > 1 else [0]
    for i, (mname, off) in enumerate(zip(models, offs)):
        g = T[T.model == mname].set_index("lang").reindex(langs)
        ax.errorbar(x + off, 100 * g.rate, yerr=[100 * (g.rate - g.ci_low), 100 * (g.ci_high - g.rate)],
                    fmt=MARKERS[i], color=SERIES[i], ms=6, lw=1.4, capsize=0, label=mname,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(x, [LANG_NAME[lang] for lang in langs])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.legend(ncol=min(k, 4), loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
    save(fig, d, name)


def fig3(H, d):
    H = H.reindex([lang for lang in LANG_ORDER if lang in H.index])
    H = H.loc[:, H.max().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * H.shape[1], 0.5 + 0.45 * H.shape[0]))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f0efec"] + SEQ)
    vmax = max(0.05, float(H.values.max()))
    im = ax.imshow(H.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.grid(False)
    nice = {k: v[2].split(" (")[0] if len(v[2]) < 28 else k.replace("_", " ").capitalize() for k, v in LABELS.items()}
    nice.update({"SHORT_RSA_KEY": "RSA key < 2048 bits", "TLS_NO_VERIFY": "TLS not verified", "WEAK_RANDOM": "Non-crypto random",
                 "WEAK_HASH": "MD5 / SHA-1", "ECB_MODE": "ECB mode", "HARDCODED_KEY": "Hard-coded key",
                 "FAST_PASSWORD_HASH": "Fast password hash", "STATIC_IV": "Static IV / nonce",
                 "LOW_KDF_ITERATIONS": "Low PBKDF2 iterations", "TIMING_COMPARE": "Non-constant-time compare",
                 "JWT_NO_VERIFY": "JWT not verified", "OLD_TLS": "Old SSL/TLS", "WEAK_CIPHER": "DES / RC4 etc.",
                 "STATIC_SALT": "Constant salt"})
    ax.set_xticks(range(H.shape[1]), [f"{nice[c]}\n({LABELS[c][0]})" for c in H.columns], rotation=40, ha="right")
    ax.set_yticks(range(H.shape[0]), [LANG_NAME[i] for i in H.index])
    for (r, c), v in np.ndenumerate(H.values):
        ax.text(c, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=7.5,
                color="#ffffff" if v > 0.55 * vmax else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=matplotlib.ticker.PercentFormatter(1.0))
    cb.outline.set_visible(False)
    ax.set_title("% of code samples with each misuse", fontsize=9, color=INK2, loc="left")
    save(fig, d, "fig3_misuse_types_heatmap")


def fig4(T, d):
    langs = _langs(T)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(langs))
    for i, (cond, off, lab) in enumerate([("baseline", -0.12, "No system prompt"), ("secure", 0.12, "Secure-coding system prompt")]):
        g = T[T.condition == cond].set_index("lang").reindex(langs)
        ax.errorbar(x + off, 100 * g.rate, yerr=[100 * (g.rate - g.ci_low), 100 * (g.ci_high - g.rate)],
                    fmt=MARKERS[i], color=SERIES[i], ms=7, lw=1.4, label=lab, markeredgecolor="white")
    for xi, lang in enumerate(langs):
        a = T[(T.condition == "baseline") & (T.lang == lang)].rate
        b = T[(T.condition == "secure") & (T.lang == lang)].rate
        if len(a) and len(b):
            ax.plot([xi - 0.12, xi + 0.12], [100 * a.iloc[0], 100 * b.iloc[0]], color=AXIS, lw=1, zorder=0)
    ax.set_xticks(x, [LANG_NAME[lang] for lang in langs])
    ax.set_ylabel("Code samples with ≥1 misuse")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8)
    save(fig, d, "fig4_secure_prompt_effect")


def main():
    cfg = load_config()
    td, fd = cfg["_out"] / "tables", out_dir(cfg, "figures")
    fig1(fd)
    fig_rate_by_model(pd.read_csv(td / "rq1_rate_by_model_lang.csv"), fd, "fig2_misuse_rate_by_language",
                      "Code samples with ≥1 misuse")
    fig3(pd.read_csv(td / "rq2_label_rate_by_lang.csv", index_col=0), fd)
    p4 = td / "rq4_rate_by_condition_lang.csv"
    if p4.exists():
        fig4(pd.read_csv(p4), fd)
    p5 = td / "rq3_pqc_rate_by_model_lang.csv"
    if p5.exists():
        fig_rate_by_model(pd.read_csv(p5), fd, "fig5_pqc_readiness",
                          "Long-term answers recommending PQC")
    print(f"Figures written to {fd}")


if __name__ == "__main__":
    main()
