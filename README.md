# Impacts of Uncertainty & Differential Privacy on Title I

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Author: [Ryan Steed](rbsteed.com), with help from Terrance Liu

Paper co-authors: Terrance Liu, [Steven Wu](https://zstevenwu.com/), [Alessandro Acquisti](https://www.heinz.cmu.edu/~acquisti/)

API documentation: [rbsteed.com/dp-policy](https://rbsteed.com/dp-policy/)

For more, check out the [paper](https://rbsteed.com/referral/dp-policy) and [SI](https://www-science-org.cmu.idm.oclc.org/doi/suppl/10.1126/science.abq4481/suppl_file/science.abq4481_sm.pdf).

## Installation

```
make dp_policy
```

## Running the CLI
Use the [CLI endpoints](https://rbsteed.com/dp-policy/api.html) in `dp_policy/api.py`.

```bash
dp_policy --help
# to run a specific experiment
dp_policy run [experiment]
# to only produce the feather file for regression analysis (using cached results)
dp_policy run --just-join [name]
# to run all experiments
dp_policy run_all
```

Experiment options (described in detail the [SI](https://www-science-org.cmu.idm.oclc.org/doi/suppl/10.1126/science.abq4481/suppl_file/science.abq4481_sm.pdf) and passed to the [`Experiment.get_experiment`](https://rbsteed.com/dp-policy/experiments.html#dp_policy.experiments.Experiment.get_experiment) factory method) include:

- `"baseline"` - just the baseline settings (no experimental modifications).
- Policy changes
    - `"hold_harmless"` - treatments which add one or both of the post-formula provisions (hold harmless and the state minimum).
    - `"thresholds"` - treatments which modify the thresholds for district funding eligibility.
    - `"moving_average"` - treatments which use multiyear averages of varying size.
    - `"budget`" - treatments which vary the overall Title I appropriation.
- Robustness checks
    - `"post_processing"` - treatments which modify the post-processing applied after noise injection.
    - `"epsilon"` - treatments which modify the privacy parameter epsilon.
    - `"sampling`" - treatments which vary the variance of simulated data error.
    - `"vary_total_children"` - treatment where the number of total children is also noised.

## Replicating Results
1. For general statistics, run cells in `notebooks/results.ipynb`.
2. Generate all the experimental results by running `dp_policy run_all` or running chosen experiments individually with `dp_policy run [experiment]`.
3. Visualize experiment results with `notebooks/policy_experiments.ipynb`. (For example, Fig. 1 was produced with statistics from the Epsilon Sensitivity section of `notebooks/policy_experiments.ipynb`.)
4. Produce disparity plots and GAM smooth plots with `R/plot_all.R`. (For example, Fig. 2 is the race disparity plot for the `"hold_harmless"` experiment.)

## Contents
- `data/`
    - `discrimination/` - [ACS 5-year data for discrimination analysis](https://nces.ed.gov/programs/edge/tableviewer/acsProfile/2019)
    - `shapefiles/` - [TIGER shapefiles for school districts](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
    - `titlei-allocations` - official dep. of ed. figures, from Todd Stephenson
    - `saipe*`, `county_saipe*` - district- and county-level [SAIPE data](https://www.census.gov/data/datasets/2020/demo/saipe/2020-school-districts.html)
    - `fips_codes.csv` - map  of FIPS codes to postal codes and state names
    - `nslp19.csv` - [National School Lunch program data](https://nces.ed.gov/ccd/files.asp#Fiscal:2,LevelId:7,SchoolYearId:34,Page:1) (exploration only)
    - `sppe*` - [state per-pupil expenditure data](https://nces.ed.gov/ccd/pub_rev_exp.asp)
- `dp_policy/` - codebase
    - `titlei/` - submodule for replicating the Title I allocation procedure, with noise
        - `allocators.py` - allocation procedures
        - `bootstrap.py` - exploratory functions for sampling experiments
        - `evaluation.py` - utility functions for evaluating results
        - `mechanisms.py` - randomization (noise injection) mechanisms
        - `thresholders.py` - thresholding mechanisms for formula
        - `utils.py` - utility functions
    - `api.py` - endpoints for CLI
    - `config.py` - settings
    - `experiments.py` - set of experiment configurations for replicating results
- `logs/` - logs for recording runs
- `notebooks/` - Jupyter notebooks for exploration and visualization
    - `results.ipynb` - main notebook for replicating and visualizing auxiliary experiment results
    - `policy-experiments.ipynb` - notebook for visualizing results of policy experiments
    - `nslp.ipynb` - exploring NSLP data as an alternative ground truth
    - `plot_sampling.ipynb` - developing sampling mechanisms
- `plots/` - output plots
- `R/` - R scripts for regression and visualization
    - `exploration.Rmd` - exploring results
    - `plot_all.R` - plots/regressions for all experiments
    - `plot_experiment.R` - plots/regressions for one experiment
    - `plots.R` - endpoints for plotting results and running regressions
    - `regression_tables.R` - endpoint for recording regression tables
    - `regressions.Rmd` - exploring regression specifications
    - `utils.R` - utility functions for plotting and regressions
- `results/` - cached results files
    - `policy_experiments/` - for experiment runs
    - `regressions/` - for regressions
- `scripts/` - miscellaneous bash scripts to make server runs easier

## Documentation
Documentation for the `dp-policy` API is published at [rbsteed.com/dp-policy](https://rbsteed.com/dp-policy).

To generate the documentation, use pdoc3:

```bash
pdoc --html --output-dir docs --force dp_policy --template-dir docs/templates
git subtree push --prefix docs/dp_policy origin gh-pages
```
