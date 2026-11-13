# 08_extensions.R
# Purpose: Callaway-Sant'Anna estimates for the four extensions built by
#          07_extensions.py (software vs owner, homogenization, data exchange,
#          SEP-1 reporting)
# Author: Caprice Culkin
# Date: October 2026
#
# Outputs: output/ext_att_simple.csv, output/ext_att_dynamic.csv,
#          output/ext_pretrend.csv, output/figures/es_ext_<design>_<outcome>.png

suppressPackageStartupMessages({
  library(did)
  library(data.table)
  library(ggplot2)
})
set.seed(20261029)
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
P <- "data/processed"

# Pre-trend test. The Wald pre-test is designed for a "varying" base period;
# with a "universal" base period (used for the event-study plots) the pre-period
# covariance is often singular and the test is unreliable. So the test is run
# on a separate varying-base fit. Also reports the largest |t| among the
# event-study pre-period points (event time -5 to -2) as a second check.
pretest <- function(d, outcome, control, es) {
  fit_v <- tryCatch(
    att_gt(yname = outcome, tname = "year", idname = "hosp_num", gname = "gvar",
           data = as.data.frame(d), control_group = control, xformla = ~1,
           allow_unbalanced_panel = TRUE, base_period = "varying",
           est_method = "dr", bstrap = FALSE),
    error = function(e) NULL)
  W <- if (!is.null(fit_v) && !is.null(fit_v$W)) as.numeric(fit_v$W)[1] else NA_real_
  p <- if (!is.null(fit_v) && !is.null(fit_v$Wpval)) as.numeric(fit_v$Wpval)[1] else NA_real_
  pre <- es$egt <= -2 & !is.na(es$se.egt) & es$se.egt > 0
  max_t <- if (any(pre)) max(abs(es$att.egt[pre] / es$se.egt[pre])) else NA_real_
  data.table(pretrend_W = W, pretrend_p = p, max_abs_pre_t = max_t)
}

# One estimation. `d` already holds only the rows for this design.
estimate <- function(d, outcome, control, design) {
  d <- d[!is.na(d[[outcome]]) & !is.na(d$gvar) & !is.na(d$hosp_num)]
  if (nrow(d) == 0 || uniqueN(d$year) < 3) {
    message(sprintf("  skip %s / %s: too few years", design, outcome)); return(NULL)
  }
  n_treated <- uniqueN(d[gvar > 0, hosp_num])
  if (n_treated < 10) {
    message(sprintf("  skip %s / %s: only %d treated hospitals", design, outcome, n_treated))
    return(NULL)
  }
  att <- tryCatch(
    att_gt(yname = outcome, tname = "year", idname = "hosp_num", gname = "gvar",
           data = as.data.frame(d), control_group = control, xformla = ~1,
           allow_unbalanced_panel = TRUE, base_period = "universal",
           est_method = "dr", bstrap = TRUE, cband = TRUE),
    error = function(e) { message("  error: ", conditionMessage(e)); NULL })
  if (is.null(att)) return(NULL)
  s <- tryCatch(aggte(att, type = "simple", na.rm = TRUE), error = function(e) NULL)
  es <- tryCatch(aggte(att, type = "dynamic", min_e = -5, max_e = 5, na.rm = TRUE),
                 error = function(e) NULL)
  if (is.null(s) || is.null(es)) {
    message(sprintf("  skip %s / %s: no valid group-time estimates to aggregate", design, outcome))
    return(NULL)
  }
  p <- ggdid(es) + labs(title = sprintf("%s: %s", design, outcome),
                        x = "Years relative to treatment", y = "ATT")
  ggsave(sprintf("output/figures/es_ext_%s_%s.png", design, outcome), p,
         width = 7, height = 4.5, dpi = 200)
  list(
    simple = data.table(design = design, outcome = outcome, control = control,
                        att = s$overall.att, se = s$overall.se,
                        ci_low = s$overall.att - 1.96 * s$overall.se,
                        ci_high = s$overall.att + 1.96 * s$overall.se,
                        n_hospitals = uniqueN(d$hosp_num), n_treated = n_treated),
    dynamic = data.table(design = design, outcome = outcome, event_time = es$egt,
                         att = es$att.egt, se = es$se.egt),
    pretrend = cbind(data.table(design = design, outcome = outcome),
                     pretest(d, outcome, control, es)))
}

results <- list()
add <- function(r) if (!is.null(r)) results[[length(results) + 1]] <<- r

# Keep one arm's treated hospitals plus never-acquired controls. Computed with
# $ outside data.table brackets so the argument is not confused with a column.
arm_rows <- function(d, arm_name) {
  (d$arm %in% c(arm_name, "never_acquired")) & !is.na(d$gvar)
}

main_outcomes <- c("sep1", "sep_shock_3hr", "sep_sev_3hr", "sep_shock_6hr", "sep_sev_6hr", "ed_time", "ed_lwbs", "staff_flu_vax", "opioid_ecqm",
                   "ed_time_psych", "stroke_ct_45min", "psi90", "psi03_ulcer", "psi13_sepsis", "psi11_resp_failure", "psi12_pe_dvt", "mort_copd", "mort_cabg", "readm_copd", "readm_cabg", "readm_hipknee", "comp_hipknee", "edac_hf", "edac_pn",
                   "mort_ami", "mort_hf", "mort_pn", "readm_ami", "readm_hf", "readm_pn",
                   "log_drg_per_mcr_dc", "drg_per_mcr_dc", "op_margin", "total_margin", "log_total_costs", "cost_per_dc", "mcr_dc_share", "vbp_tps", "log_charges_per_dc", "log_markup", "log_npr_per_dc", "discount_share", "fte_per_bed", "log_total_dc", "log_cost_per_dc", "outlier_per_mcr_dc", "outlier_share", "bad_debt_share", "charity_share")

# ---- EXT 1: software vs owner ----
for (g in c("switch_only", "acq_and_switch", "acq_no_switch")) {
  f <- file.path(P, sprintf("ext1_%s.csv", g))
  if (!file.exists(f)) next
  d <- fread(f)
  for (y in intersect(main_outcomes, names(d))) {
    message(sprintf("ext1 | %s | %s", g, y))
    add(estimate(d, y, "nevertreated", paste0("ext1_", g)))
  }
}

# ---- EXT 2: homogenization (acquired hospitals only, not-yet-acquired as comparison) ----
f <- file.path(P, "ext2_converge.csv")
if (file.exists(f)) {
  d <- fread(f)
  for (y in grep("^dist_", names(d), value = TRUE)) {
    message(sprintf("ext2 | converge | %s", y))
    add(estimate(d, y, "notyettreated", "ext2_converge"))
  }
}

# ---- EXT 3: data exchange ----
f <- file.path(P, "ext3_exchange.csv")
if (file.exists(f)) {
  d <- fread(f)
  ex_vars <- setdiff(names(d), c("id", "hosp_num", "year", "arm", "event_type", "gvar"))
  for (a in c("epic", "oracle")) {
    da <- d[arm_rows(d, a)]
    for (y in ex_vars) {
      message(sprintf("ext3 | %s | %s", a, y))
      add(estimate(da, y, "nevertreated", paste0("ext3_", a)))
    }
  }
}

# ---- EXT 4: SEP-1 reporting ----
f <- file.path(P, "ext4_reporting.csv")
if (file.exists(f)) {
  d <- fread(f)
  for (a in c("epic", "oracle")) {
    da <- d[arm_rows(d, a)]
    for (y in intersect(c("sep1_reported", "sep1_log_cases"), names(d))) {
      message(sprintf("ext4 | %s | %s", a, y))
      add(estimate(da, y, "nevertreated", paste0("ext4_", a)))
    }
  }
}

simple <- rbindlist(lapply(results, `[[`, "simple"))
fwrite(simple, "output/ext_att_simple.csv")
fwrite(rbindlist(lapply(results, `[[`, "dynamic")), "output/ext_att_dynamic.csv")
fwrite(rbindlist(lapply(results, `[[`, "pretrend"), fill = TRUE), "output/ext_pretrend.csv")

cat("\nOverall ATT by extension:\n")
print(simple[, .(design, outcome, att = round(att, 3), ci_low = round(ci_low, 3),
                 ci_high = round(ci_high, 3), n_treated)], nrows = 500)
cat("\nPre-trend checks (want large p and max |t| below about 2):\n")
print(rbindlist(lapply(results, `[[`, "pretrend"), fill = TRUE), nrows = 500)
cat("\nSaved output/ext_*.csv and output/figures/es_ext_*.png\n")
