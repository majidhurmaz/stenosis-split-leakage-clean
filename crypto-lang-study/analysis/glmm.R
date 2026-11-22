# Frequentist GLMM for RQ1 and RQ4 (the model named in the paper guide).
# Needs: install.packages(c("lme4", "emmeans", "readr", "dplyr"))
# Run from the project folder:  Rscript analysis/glmm.R
# NOTE: written to match src/stats.py but not run by the author of the pipeline (no R available);
# check the output against results/tables/rq1_odds_ratios_vs_english.csv - the ORs should be close.
suppressPackageStartupMessages({library(lme4); library(emmeans); library(readr); library(dplyr)})

s <- read_csv("results/scores.csv", show_col_types = FALSE) |>
  filter(status == "code") |>
  mutate(lang = factor(lang, levels = c("en", "ur", "ru", "cm", "zh", "ar")),
         misuse_any = as.integer(misuse_any),
         secure = as.integer(condition == "secure"))
dir.create("results/tables", showWarnings = FALSE)

# RQ1: language fixed effect, random intercepts for task and model (baseline condition only)
base <- filter(s, condition == "baseline")
m1 <- glmer(misuse_any ~ lang + (1 | task) + (1 | model), data = base, family = binomial,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
print(summary(m1))
ci <- confint(m1, parm = "beta_", method = "Wald")
or <- data.frame(term = names(fixef(m1)), OR = exp(fixef(m1)), CI_low = exp(ci[, 1]), CI_high = exp(ci[, 2]))
write_csv(or, "results/tables/rq1_glmm_odds_ratios.csv")
m0 <- update(m1, . ~ . - lang)
print(anova(m0, m1))  # likelihood-ratio test for the language effect
pw <- pairs(emmeans(m1, ~ lang), adjust = "holm", type = "response")
write_csv(as.data.frame(pw), "results/tables/rq1_glmm_pairwise_holm.csv")

# RQ4: language x secure-prompt interaction
m4 <- glmer(misuse_any ~ lang * secure + (1 | task) + (1 | model), data = s, family = binomial,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
m4a <- update(m4, . ~ . - lang:secure)
print(anova(m4a, m4))  # does the secure prompt change the language gap?
eff <- contrast(emmeans(m4, ~ secure | lang), method = "revpairwise", adjust = "holm", type = "response")
write_csv(as.data.frame(confint(eff)), "results/tables/rq4_glmm_secure_effect_by_lang.csv")
