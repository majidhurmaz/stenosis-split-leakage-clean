# Prompt language aur LLM ke cryptographic code ki security — study pipeline

Paper: *Lost in Translation, Weak in Cryptography: How the Prompt Language Affects the
Cryptographic Security of LLM-Generated Code* (target: Cryptographic Theory and Practice, Springer).

Yeh folder poori study chalata hai: 20 crypto tasks × 6 zabanen → LLMs → code nikalna →
3 engines se scoring → manual review → statistics → figures/tables.

## Folder

| Path | Kya hai |
|---|---|
| `data/prompts.json` | 20 tasks, har aik 6 zabanon mein (en, ur, ru = Roman Urdu, cm = code-mixed, zh, ar) + secure system prompt |
| `config.yaml` | models, repeats (5), temperature (0.7) |
| `src/generate.py` | models ko prompts bhejta hai → `results/raw/` (ruk jaye to dobara chalayen, wahin se shuru hoga) |
| `src/extract.py` | jawab se Python code nikalta hai → `results/code/`, `results/samples.csv` |
| `src/detectors.py` | apne AST detectors (14 misuse types, har aik CWE se jura) |
| `rules/crypto.yaml` | Semgrep ke local crypto rules (internet nahi chahiye) |
| `src/score.py` | custom + Bandit + Semgrep → `results/findings.csv`, `results/scores.csv` |
| `src/pqc.py` | post-quantum score (RQ3) → `results/pqc.csv` |
| `src/manual.py` | 20% manual review sheets, Cohen kappa, tool vs manual (Table 5) |
| `src/stats.py` | RQ1–RQ4 statistics → `results/tables/` |
| `src/figures.py` | Fig. 1–5 (PNG 300 dpi + PDF) → `results/figures/` |
| `analysis/glmm.R` | lme4 mixed model (agar R ho) |
| `tests/test_detectors.py` | 30 known-answer tests detectors ke liye |

## Pehle yeh 3 kaam (data collection se pehle)

1. **Translations check karwayen.** `prompts.json` mein Urdu, Roman Urdu aur code-mixed aap khud
   dekhen; Chinese aur Arabic kisi native speaker se. Har zaban ka doosra banda back-translate kare
   (English mein wapis) aur meaning compare ho. Jo badlen, `prompts.json` mein hi badlen.
   Dhyan: prompts mein "secure / mehfooz / محفوظ" jaise alfaz na aayen (sirf T19, T20 mein waqt
   wali shart jaan-boojh kar hai).
2. **Ollama install karein** (ollama.com), phir:
   ```
   ollama pull qwen2.5-coder:7b-instruct-q4_K_M
   ollama pull llama3.1:8b-instruct-q4_K_M
   ollama pull deepseek-coder:6.7b-instruct-q4_K_M
   ollama list
   ```
   `ollama list` mein jo exact tag aaye wahi `config.yaml` mein likhen. Commercial model ke liye
   config mein comment hata kar API key environment variable mein rakhen (key file mein na likhen).
3. **`config.yaml` se `mock` model hata dein** (woh sirf testing ke liye hai).

## Chalane ka tareeqa (Windows, PowerShell)

```
cd crypto-lang-study
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tests\test_detectors.py            # "all 30 detector cases pass" aana chahiye

python src\generate.py --dry-run           # kitni calls hongi
python src\generate.py                     # asal run (GPU par kai ghante; raat bhar chalayen)
python src\extract.py
python src\score.py                        # Semgrep na chale to: python src\score.py --no-semgrep
python src\pqc.py
python src\manual.py sample                # rater_A.csv aur rater_B.csv bante hain
#   ... dono reviewers alag alag 0/1 bharen ...
python src\manual.py agree                 # kappa + disagreements.csv
#   ... disagreements.csv mein final_* columns mil kar bharen ...
python src\manual.py final                 # consensus + tool vs manual (Table 5)
python src\stats.py
python src\figures.py
```

Semgrep Windows par native support beta mein hai; agar install na ho to WSL mein chalayen ya
`--no-semgrep` istemal karein aur paper mein likhen ke do engines (custom + Bandit) istemal huay.

Real run ka hajm: 3 open models × 2 conditions × 6 zabanen × 20 tasks × 5 = 3,600 calls.
Aik commercial model add karein to +1,200.

## Output → paper

| Paper item | File |
|---|---|
| Fig. 1 pipeline | `figures/fig1_pipeline` |
| Fig. 2 misuse rate by language | `figures/fig2_misuse_rate_by_language` |
| Fig. 3 misuse types heatmap | `figures/fig3_misuse_types_heatmap` |
| Fig. 4 secure prompt | `figures/fig4_secure_prompt_effect` |
| Fig. 5 PQC readiness | `figures/fig5_pqc_readiness` |
| Table 1 tasks | `data/prompts.json` (id, category, target_cwes, en) |
| Table 3 rubric | `results/manual/rubric.csv` |
| Table 4 main results | `tables/table4_main_results.csv` |
| Table 5 agreement | `manual/table5a_inter_rater.csv`, `manual/table5b_tool_vs_manual.csv` |
| RQ1 | `tables/rq1_*` (OR vs English, Holm pairwise, Wald test) |
| RQ2 | `tables/rq2_*` |
| RQ3 | `tables/rq3_*` (long-term tasks T19, T20; T07–T09 control) |
| RQ4 | `tables/rq4_*` |
| Supplementary S1 | `tables/tableS1_answer_status.csv` (refusal / no-code) |

## Methods mein likhne wale design faisle

- **Primary outcome:** code wale jawab mein kam az kam aik "misuse" (14 labels, `src/detectors.py`
  mein `severity = "misuse"`). "weak" labels (CBC baghair MAC, unencrypted private key, PKCS#1 v1.5,
  pycrypto, Bandit ke demo passwords) alag report hote hain, primary mein nahi gine jate.
- Label tab lagta hai jab teeno engines mein se **koi bhi** use pakre (union); har engine ka
  alag rate bhi `scores.csv` mein hai.
- Thresholds: PBKDF2-HMAC-SHA256 ≥ 600,000 aur SHA-512 ≥ 210,000 iterations (OWASP Password
  Storage Cheat Sheet), RSA ≥ 2048 bits (NIST SP 800-131A). Paper mein jo version aap ne dekha
  uski tareekh likhen.
- **Statistics:** logistic regression, language (ref = English) + model fixed effects,
  task par cluster-robust SE; sensitivity ke liye Bayesian mixed model (task + model random
  intercepts) aur `analysis/glmm.R` (lme4). Pairwise par Holm correction.
- Refusal / no-code jawab misuse analysis se bahar hain, magar unki shareh har zaban ke liye report hoti hai.
- Secure system prompt English mein hai, har zaban ke liye wahi (yeh threats to validity mein likhen).
- Manual review blind hai: sheet mein zaban aur model nazar nahi aate (code comments mein zaban
  dikh sakti hai — yeh bhi limitation mein likhen).

## Pipeline test

Is folder ko `mock` model (canned jawab) se poora chala kar test kiya gaya hai. Mock ke numbers
**asal data nahi** — `results/` folder asal run se pehle delete kar dein.
