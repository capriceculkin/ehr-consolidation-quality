# 03_analysis.R
# Purpose: Callaway-Sant'Anna staggered difference-in-differences estimates
#          of Epic and Oracle system acquisition on CMS quality measures
# Author: Caprice Culkin
# Date: October 2026
#
# Input:  data/processed/analysis_panel.csv (06_build_analysis_panel.py)
# Output: output/att_simple.csv      overall ATT per arm x outcome
#         output/att_dynamic.csv     event-study estimates per arm x outcome
#         output/pretrend_tests.csv  pre-period Wald test per arm x outcome
#         output/figures/es_<arm>_<outcome>.png
#
# Main specification
#   - Treated: hospitals acquired into an Epic-standardized (or Oracle-
#     standardized) system, timed by acquisition year (gvar)
#   - Comparison: never-acquired hospitals (gvar = 0). The not-yet-treated
#     comparison is run as a sensitivity check.
#   - Unconditional parallel trends (xformla = ~1) as the main model;
#     bed size as a covariate in the sensitivity model
#   - Standard errors: multiplier bootstrap clustered by hospital
#
# Install once:  install.packages(c("did", "data.table", "ggplot2"))

suppressPackageStartupMessages({
  library(did)
  library(data.table)
  library(ggplot2)
})
set.seed(20261029)

dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)

panel <- fread("data/processed/analysis_panel.csv")

OUTCOMES <- c(
  # documentation-dependent
  "sep1",
  # SEP-1 components, ED workflow, eCQM, and a placebo (staff flu vaccination)
  "sep_shock_3hr", "sep_sev_3hr", "sep_shock_6hr", "sep_sev_6hr", "ed_time", "ed_lwbs", "staff_flu_vax", "opioid_ecqm",
  "ed_time_psych", "stroke_ct_45min", "psi90", "psi03_ulcer", "psi13_sepsis", "psi11_resp_failure", "psi12_pe_dvt", "mort_copd", "mort_cabg", "readm_copd", "readm_cabg", "readm_hipknee", "comp_hipknee", "edac_hf", "edac_pn",
  # clinically driven (claims-based)
  "mort_ami", "mort_hf", "mort_pn", "readm_ami", "readm_hf", "readm_pn", "readm_hw",
  # payment, cost and margin (Medicare cost reports) and VBP score (Aim 3)
  "log_drg_per_mcr_dc", "drg_per_mcr_dc", "op_margin", "total_margin", "log_total_costs", "cost_per_dc", "mcr_dc_share", "vbp_tps", "log_charges_per_dc", "log_markup", "log_npr_per_dc", "discount_share", "fte_per_bed", "log_total_dc", "log_cost_per_dc", "outlier_per_mcr_dc", "outlier_share", "bad_debt_share", "charity_share"
)
OUTCOMES <- intersect(OUTCOMES, names(panel))
ARMS <- c("epic", "oracle")

# Rows for one arm: that arm's treated hospitals plus never-acquired.
# Mixed and unclassified acquisitions (gvar NA) are excluded.
# NOTE: this is computed OUTSIDE panel[...] on purpose. Inside data.table
# brackets, a name like `arm` resolves to the column, not to a function
# argument, which would silently keep both arms.
arm_keep <- function(arm_name) {
  (panel$arm %in% c(arm_name, "never_acquired")) & !is.na(panel$gvar)
}

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

# -------------------------
# One estimation run
# -------------------------
run_cs <- function(arm_name, outcome, control = "nevertreated", xformla = ~1) {
  keep <- arm_keep(arm_name)
  d <- panel[keep]
  d <- d[!is.na(d[[outcome]])]
  treated_arms <- unique(d[gvar > 0, arm])
  stopifnot(length(treated_arms) == 0 || identical(treated_arms, arm_name))
  if (control == "nevertreated" && !any(d$gvar == 0)) return(NULL)
  if (uniqueN(d$year) < 3) {
    message(sprintf("  skip %s / %s: outcome observed in fewer than 3 years", arm_name, outcome))
    return(NULL)
  }
  n_treated <- uniqueN(d[gvar > 0, hosp_num])
  if (n_treated < 10) {
    message(sprintf("  skip %s / %s: only %d treated hospitals with data",
                    arm_name, outcome, n_treated))
    return(NULL)
  }
  if (length(all.vars(xformla)) > 0) d <- d[!is.na(bdtot)]
  att <- tryCatch(
    att_gt(yname = outcome, tname = "year", idname = "hosp_num", gname = "gvar",
           data = as.data.frame(d), control_group = control, xformla = xformla,
           allow_unbalanced_panel = TRUE, base_period = "universal",
           est_method = "dr", bstrap = TRUE, cband = TRUE),
    error = function(e) { message("  error: ", conditionMessage(e)); NULL })
  if (is.null(att)) return(NULL)
  list(att = att, data = d, n_hospitals = uniqueN(d$hosp_num), n_treated = n_treated)
}

# -------------------------
# Coverage before estimating
# -------------------------
cat("\nTreated hospitals by arm and acquisition year:\n")
print(dcast(unique(panel[arm %in% ARMS, .(id, arm, gvar)])[, .N, by = .(arm, gvar)],
            gvar ~ arm, value.var = "N", fill = 0))
cat("\nNon-missing hospital-years per outcome and year:\n")
print(panel[, lapply(.SD, function(x) sum(!is.na(x))), by = year, .SDcols = OUTCOMES][order(year)])

# -------------------------
# Estimate
# -------------------------
simple <- list(); dynamic <- list(); pre_rows <- list()

specs <- list(
  main         = list(control = "nevertreated", xformla = ~1),
  notyettreated = list(control = "notyettreated", xformla = ~1),
  beds          = list(control = "nevertreated", xformla = ~bdtot)
)

for (spec_name in names(specs)) {
  sp <- specs[[spec_name]]
  for (a in ARMS) for (y in OUTCOMES) {
    message(sprintf("%s | %s | %s", spec_name, a, y))
    fit <- run_cs(a, y, sp$control, sp$xformla)
    if (is.null(fit)) next
    att <- fit$att

    s <- tryCatch(aggte(att, type = "simple", na.rm = TRUE), error = function(e) NULL)
    es_check <- tryCatch(aggte(att, type = "dynamic", min_e = -5, max_e = 5, na.rm = TRUE),
                         error = function(e) NULL)
    if (is.null(s) || is.null(es_check)) {
      message(sprintf("  skip %s / %s: no valid group-time estimates to aggregate", a, y))
      next
    }
    simple[[length(simple) + 1]] <- data.table(
      spec = spec_name, arm = a, outcome = y, att = s$overall.att,
      se = s$overall.se, ci_low = s$overall.att - 1.96 * s$overall.se,
      ci_high = s$overall.att + 1.96 * s$overall.se,
      n_hospitals = fit$n_hospitals, n_treated = fit$n_treated)

    pre_rows[[length(pre_rows) + 1]] <- cbind(
      data.table(spec = spec_name, arm = a, outcome = y),
      pretest(fit$data, y, sp$control, es_check))

    if (spec_name == "main") {
      es <- es_check
      dynamic[[length(dynamic) + 1]] <- data.table(
        arm = a, outcome = y, event_time = es$egt, att = es$att.egt,
        se = es$se.egt, crit_val = es$crit.val.egt)
      p <- ggdid(es) + labs(title = sprintf("%s acquisition: %s", a, y),
                            x = "Years relative to acquisition", y = "ATT")
      ggsave(sprintf("output/figures/es_%s_%s.png", a, y), p, width = 7, height = 4.5, dpi = 200)
    }
  }
}

fwrite(rbindlist(simple), "output/att_simple.csv")
fwrite(rbindlist(dynamic), "output/att_dynamic.csv")
if (length(pre_rows)) {
  fwrite(rbindlist(pre_rows, fill = TRUE), "output/pretrend_tests.csv")
  cat("\nPre-trend checks, main specification (want large p and max |t| below about 2):\n")
  print(rbindlist(pre_rows, fill = TRUE)[spec == "main"])
}

cat("\nOverall ATT, main specification:\n")
print(rbindlist(simple)[spec == "main"])
cat("\nSaved results to output/\n")

# -------------------------
# Core thesis comparison (next step, not run here)
# -------------------------
# The thesis question is whether the Epic effect on SEP-1 is larger than the
# effect on clinically driven outcomes. Comparing the two ATTs needs their
# covariance, since they come from the same hospitals. Plan with the
# advisor: stack outcomes after standardizing each, or bootstrap the
# difference by resampling hospitals and re-running both estimates.
