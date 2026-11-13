# Ownership, software, and what Medicare pays for

Code for my MS thesis at the Yale School of Public Health (advisor: Lee Kennedy-Shaffer).

When a hospital is bought by a health system, it usually moves onto that system's electronic health record, most often Epic. Medicare then judges and pays the hospital partly on measures that are produced through that record. This project asks two questions:

1. **Aim 1.** When a hospital is acquired into a system that runs Epic or Oracle Health, do documentation-dependent quality measures like SEP-1 (the sepsis bundle) move differently from claims-based patient outcomes like 30-day mortality and readmission?
2. **Aim 2.** Is the effect coming from the new owner or from the new software? Hospitals that switched to Epic without being acquired are compared with hospitals acquired with and without a software switch, on quality measures and on prices and Medicare payment.

The estimator is Callaway and Sant'Anna (2021) staggered difference-in-differences, because hospitals are acquired in different years and two-way fixed effects can be biased when treatment timing varies.

## Data

The real analysis links the AHA Annual Survey and AHA IT Supplement (licensed through WRDS), CMS Care Compare, Medicare cost reports, Hospital VBP files, and the AHRQ Compendium of U.S. Health Systems, 2008 to 2024.

**None of that data is in this repo.** The AHA files are licensed and cannot be shared. Instead, `code/00_make_fake_panel.py` builds a fake panel with the same columns, so anyone can run the descriptives and the estimator.

## Try it on fake data

```bash
pip install pandas numpy matplotlib
python code/00_make_fake_panel.py      # fake hospital-year panel
python code/09_descriptives.py         # timing, attrition, baseline table, trend plots
python code/check_cs_estimates.py sep1 mort_ami   # Callaway-Sant'Anna in Python
```

The fake panel has two effects built in on purpose. For the Epic arm, SEP-1 drops 3 points in the year of acquisition and the year after. For the Oracle arm, it drops 2 points. Mortality has no effect. If the estimator works, the event study should show roughly -3 at event years 0 and 1 for Epic, and nothing for mortality. With the default seed it gives -3.5 and -3.6 for Epic SEP-1, and mortality estimates near zero.

The R version of the estimator (`code/03_analysis.R`, using the `did` package) reads the same file:

```r
install.packages(c("did", "data.table", "ggplot2"))
source("code/03_analysis.R")
```

## How the real pipeline runs

Steps 1 to 5 need the licensed data, so they will not run here. They are included so the full process is visible.

1. `01_data_pull.py`, `01b_compendium_systems.py`, `01c_pull_it_exchange.py` pull the AHA IT Supplement from WRDS and the AHRQ system files. Your WRDS username and password are asked for when the script runs.
2. `02_merge_v2.py` merges the AHA files on hospital ID and year, defines an acquisition as a change in health system ID, and labels each acquiring system Epic, Oracle, or mixed by the vendors of its member hospitals. `check_cohorts.py` rebuilds the same groups with separate code and must match.
3. `05_stack_cms.py`, `05b_build_cost_reports.py`, `05c_stack_hvbp.py` (with `03_download_hvbp.py`) build the quality, cost report and VBP outcomes. `check_cost_reports.py` rebuilds the payment outcome a second way.
4. `06_build_analysis_panel.py` attaches every outcome and builds `gvar`, the acquisition year used by the estimator.
5. `03_analysis.R` estimates the main effects and event studies. `07_extensions.py` and `08_extensions.R` build and estimate the Aim 2 comparison groups.
6. `check_cs_estimates.py` is an independent Python version of the estimator. It must reproduce the R estimates.
7. `09_descriptives.py` makes the descriptive tables and plots.
8. Secondary analyses: `11_contagion.py` (does Epic spread through a hospital's own system or its local market), `12_safety_net_gradient.py` (do effects differ for safety-net hospitals), `13_measure_validity.py` (does a hospital's SEP-1 move with its mortality year to year), and `check_vendor_timing.py` (AHA vendor timing against CMS attestation records).

`09_cea_bounds.py` and `10_cea_psa.py` are a cost-effectiveness analysis from an earlier version of the thesis plan. They are kept for reference but are not part of the current thesis.

## Key definitions

- **Acquisition year:** the first year a hospital's health system ID changes to a new system, or from none to a system. A hospital's first observed year never counts.
- **Epic or Oracle system:** at least 75% of the acquiring system's other hospitals used that vendor the year before the acquisition, with at least 2 hospitals reporting a vendor. Otherwise the system is "mixed."
- **Treated hospitals:** an independent hospital joining an existing system, or a hospital moving on its own into one. Whole-system mergers and newly created system IDs are left out of the main analysis.
- **Comparison group:** hospitals never acquired during the study period.

Settings are in `code/config.py`.

## Author

Caprice Culkin, MS Health Informatics, Yale School of Public Health
