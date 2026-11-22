"""Step 8: statistics for RQ1-RQ4. Writes CSV tables to results/tables/.

Primary model (RQ1): logistic regression of misuse_any on language (reference = English)
with model as a fixed effect and standard errors clustered by task (cluster-robust, 20 clusters).
Sensitivity: Bayesian mixed logistic model with random intercepts for task and model
(statsmodels BinomialBayesMixedGLM). analysis/glmm.R fits the frequentist GLMM (lme4) if you
have R; report whichever you name in the Methods, and the other as a sensitivity check.

Only answers that contain code are analysed for misuse; refusal / no-code rates are reported
separately (Table S1).
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(__file__))
from common import load_config, out_dir  # noqa: E402
from detectors import LABELS  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
LANG_ORDER = ["en", "ur", "ru", "cm", "zh", "ar"]
LANG_NAME = {"en": "English", "ur": "Urdu", "ru": "Roman Urdu", "cm": "Code-mixed", "zh": "Chinese", "ar": "Arabic"}
MISUSE = [k for k, v in LABELS.items() if v[1] == "misuse"]


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return lo, hi


def rate_table(df, by, col):
    g = df.groupby(by)[col].agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    g[["ci_low", "ci_high"]] = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    return g.rename(columns={"sum": "k", "count": "n"})


def logit_cluster(formula, df, cluster):
    m = smf.logit(formula, data=df).fit(disp=False, cov_type="cluster",
                                        cov_kwds={"groups": pd.factorize(df[cluster])[0]})
    return m


def or_table(m, prefix):
    rows = []
    ci = m.conf_int()
    for name in m.params.index:
        if name.startswith(prefix):
            rows.append({"term": name, "OR": np.exp(m.params[name]), "CI_low": np.exp(ci.loc[name, 0]),
                         "CI_high": np.exp(ci.loc[name, 1]), "p": m.pvalues[name]})
    return pd.DataFrame(rows)


def pairwise_lang(m, langs):
    """All pairwise language contrasts from a fitted model, Holm-adjusted."""
    def coef(lang):
        return f"C(lang, Treatment('en'))[T.{lang}]" if lang != "en" else None
    rows = []
    names = list(m.params.index)
    for a, b in itertools.combinations(langs, 2):
        L = np.zeros(len(names))
        if coef(a):
            L[names.index(coef(a))] -= 1
        if coef(b):
            L[names.index(coef(b))] += 1
        t = m.t_test(L)
        est, se = float(np.ravel(t.effect)[0]), float(np.ravel(t.sd)[0])
        rows.append({"lang_a": a, "lang_b": b, "OR_b_vs_a": np.exp(est),
                     "CI_low": np.exp(est - 1.96 * se), "CI_high": np.exp(est + 1.96 * se), "p": float(np.ravel(t.pvalue)[0])})
    T = pd.DataFrame(rows)
    T["p_holm"] = multipletests(T.p, method="holm")[1]
    return T


def main():
    cfg = load_config()
    td = out_dir(cfg, "tables")
    S = pd.read_csv(cfg["_out"] / "scores.csv")
    langs = [lang for lang in LANG_ORDER if lang in S.lang.unique()]
    S["lang"] = pd.Categorical(S.lang, categories=langs)

    # ---- Table S1: what kind of answer did we get?
    st1 = S.groupby(["condition", "lang"], observed=True).status.value_counts(normalize=True).unstack(fill_value=0)
    st1.round(4).to_csv(td / "tableS1_answer_status.csv")

    base = S[(S.condition == "baseline") & (S.status == "code")].copy()
    base["misuse_any"] = base.misuse_any.astype(int)

    # ---- RQ1 descriptive (Table 4 / Fig 2)
    rate_table(base, ["lang"], "misuse_any").to_csv(td / "rq1_rate_by_lang.csv", index=False)
    rate_table(base, ["model", "lang"], "misuse_any").to_csv(td / "rq1_rate_by_model_lang.csv", index=False)

    # ---- RQ1 inferential
    f = "misuse_any ~ C(lang, Treatment('en')) + C(model)"
    m1 = logit_cluster(f, base, "task")
    or1 = or_table(m1, "C(lang")
    or1["term"] = or1.term.str.extract(r"T\.(\w+)")[0].map(LANG_NAME)
    or1.to_csv(td / "rq1_odds_ratios_vs_english.csv", index=False)
    pw = pairwise_lang(m1, langs)
    pw.to_csv(td / "rq1_pairwise_holm.csv", index=False)
    wald = m1.wald_test(" , ".join(f"C(lang, Treatment('en'))[T.{x}] = 0" for x in langs if x != "en"),
                        scalar=True)
    with open(td / "rq1_model_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(str(m1.summary()))
        fh.write(f"\n\nJoint Wald test, all language terms = 0: chi2={float(wald.statistic):.3f}, "
                 f"df={int(wald.df_denom) if hasattr(wald, 'df_denom') else len(langs) - 1}, p={float(wald.pvalue):.4g}\n")
    print("RQ1 odds ratios vs English (cluster-robust by task):")
    print(or1.round(3).to_string(index=False))
    print(f"Joint Wald p = {float(wald.pvalue):.4g}")

    # Sensitivity: Bayesian mixed model, random intercepts task + model
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        bm = BinomialBayesMixedGLM.from_formula(
            "misuse_any ~ C(lang, Treatment('en'))", {"task": "0 + C(task)", "model": "0 + C(model)"}, base
        ).fit_vb()
        sm = bm.summary().tables[0]
        sm.to_csv(td / "rq1_bayes_mixed_sensitivity.csv")
    except Exception as e:  # noqa: BLE001
        print("Bayesian mixed model skipped:", e)

    # ---- RQ2: misuse types by language (Fig 3 heatmap)
    present = [lab for lab in MISUSE if base[lab].sum() > 0]
    heat = base.groupby("lang", observed=True)[present].mean()
    heat.round(4).to_csv(td / "rq2_label_rate_by_lang.csv")
    rows = []
    for lab in present:
        ct = pd.crosstab(base.lang, base[lab].astype(int))
        if ct.shape[1] == 2:
            chi2, p, dof, _ = st.chi2_contingency(ct)
            rows.append({"label": lab, "cwe": LABELS[lab][0], "chi2": chi2, "df": dof, "p": p})
    if rows:
        R2 = pd.DataFrame(rows)
        R2["p_holm"] = multipletests(R2.p, method="holm")[1]
        R2.to_csv(td / "rq2_chi2_by_label.csv", index=False)

    # ---- RQ3: PQC readiness on long-term tasks
    pq_path = cfg["_out"] / "pqc.csv"
    if pq_path.exists():
        Q = pd.read_csv(pq_path)
        lt = Q[(Q.long_term) & (Q.condition == "baseline")]
        rate_table(lt, ["lang"], "pqc_any").to_csv(td / "rq3_pqc_rate_by_lang.csv", index=False)
        rate_table(lt, ["model", "lang"], "pqc_any").to_csv(td / "rq3_pqc_rate_by_model_lang.csv", index=False)
        rate_table(lt, ["lang"], "quantum_threat_mentioned").to_csv(td / "rq3_quantum_threat_by_lang.csv", index=False)
        ctrl = Q[(~Q.long_term) & Q.task.isin(["T07", "T08", "T09"]) & (Q.condition == "baseline")]
        rate_table(ctrl, ["lang"], "pqc_any").to_csv(td / "rq3_pqc_rate_control_tasks.csv", index=False)
        en = lt[lt.lang == "en"].pqc_any
        rows = []
        for lang in langs:
            if lang == "en":
                continue
            x = lt[lt.lang == lang].pqc_any
            tab = [[en.sum(), len(en) - en.sum()], [x.sum(), len(x) - x.sum()]]
            orr, p = st.fisher_exact(tab)
            rows.append({"lang": LANG_NAME[lang], "pqc_rate_en": en.mean(), "pqc_rate_lang": x.mean(),
                         "fisher_OR_en_vs_lang": orr, "p": p})
        if rows:
            R3 = pd.DataFrame(rows)
            R3["p_holm"] = multipletests(R3.p, method="holm")[1]
            R3.to_csv(td / "rq3_fisher_vs_english.csv", index=False)
            print("\nRQ3 PQC (long-term tasks):")
            print(R3.round(3).to_string(index=False))

    # ---- RQ4: does the secure system prompt shrink the gap?
    both = S[S.status == "code"].copy()
    if both.condition.nunique() == 2:
        both["secure"] = (both.condition == "secure").astype(int)
        both["misuse_any"] = both.misuse_any.astype(int)
        rate_table(both, ["condition", "lang"], "misuse_any").to_csv(td / "rq4_rate_by_condition_lang.csv", index=False)
        m4 = logit_cluster("misuse_any ~ C(lang, Treatment('en')) * secure + C(model)", both, "task")
        inter = [n for n in m4.params.index if ":secure" in n]
        w4 = m4.wald_test(" , ".join(f"{n} = 0" for n in inter), scalar=True)
        rows = []
        for lang in langs:
            L = np.zeros(len(m4.params))
            L[list(m4.params.index).index("secure")] = 1
            if lang != "en":
                L[list(m4.params.index).index(f"C(lang, Treatment('en'))[T.{lang}]:secure")] = 1
            t = m4.t_test(L)
            e, se = float(np.ravel(t.effect)[0]), float(np.ravel(t.sd)[0])
            rows.append({"lang": LANG_NAME[lang], "OR_secure_vs_baseline": np.exp(e),
                         "CI_low": np.exp(e - 1.96 * se), "CI_high": np.exp(e + 1.96 * se), "p": float(np.ravel(t.pvalue)[0])})
        R4 = pd.DataFrame(rows)
        R4["p_holm"] = multipletests(R4.p, method="holm")[1]
        R4.to_csv(td / "rq4_secure_prompt_effect_by_lang.csv", index=False)
        with open(td / "rq4_model_summary.txt", "w", encoding="utf-8") as fh:
            fh.write(str(m4.summary()))
            fh.write(f"\n\nJoint Wald test, language x secure interaction = 0: p={float(w4.pvalue):.4g}\n")
        print("\nRQ4 secure-prompt effect by language:")
        print(R4.round(3).to_string(index=False))
        print(f"Interaction Wald p = {float(w4.pvalue):.4g}")

    # ---- Table 4: misuse % per language x model, with OR vs English
    t4 = rate_table(base, ["model", "lang"], "misuse_any")
    t4["cell"] = [f"{100 * r:.1f} ({100 * lo:.1f}-{100 * hi:.1f})" for r, lo, hi in zip(t4.rate, t4.ci_low, t4.ci_high)]
    T4 = t4.pivot(index="model", columns="lang", values="cell")
    T4.columns = [LANG_NAME[c] for c in T4.columns]
    allrow = rate_table(base, ["lang"], "misuse_any")
    T4.loc["All models"] = [f"{100 * r:.1f} ({100 * lo:.1f}-{100 * hi:.1f})"
                            for r, lo, hi in zip(allrow.rate, allrow.ci_low, allrow.ci_high)]
    orrow = ["1 (ref)"] + [f"{o:.2f} ({lo:.2f}-{hi:.2f})" for o, lo, hi in zip(or1.OR, or1.CI_low, or1.CI_high)]
    T4.loc["Adjusted OR vs English"] = orrow
    T4.to_csv(td / "table4_main_results.csv")
    print(f"\nTables written to {td}")


if __name__ == "__main__":
    main()
