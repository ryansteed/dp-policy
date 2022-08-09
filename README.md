# DP Policy Decision-Making

Author: Ryan Steed

WIP

## Installation

```
make dp_policy
```

## Running the CLI
CLI endpoints in `dp_policy/api.py`.

```bash
dp_policy --help
# to run a specific experiment
dp_policy run [experiment]
# to only produce the feather file for regression analysis (using cached results)
dp_policy run --just-join [name]
# to run all experiments
dp_policy run_all
```

## Replicating Results
1. For general results, run cells in `notebooks/results.ipynb`.
2. Experimental results.
  1. Generate all the experimental results by running `dp_policy run_all` or running chosen experiments sindividually with `dp_policy run [experiment]`.
  2. Visualize experiment results with `notebooks/policy_experiments.ipynb`. (For example, Fig. 1 was produced with statistics from the Epsilon Sensitivity section of `notebooks/policy_experiments.ipynb`.)
  3. Produce disparity plots and GAM smooth plots with `R/plot_all.R`. (For example, Fig. 2 is the race disparity plot for the `hold_harmless` experiment.)

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
    - `mechanisms.py` - randomization mechanisms
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

## Administration
### Documentation
Documentation for the `dp-policy` API is published at [rbsteed.com/dp-policy](https://rbsteed.com/dp-policy).

To generate the documentation, use pdoc3:

```bash
pdoc3 --html --output-dir docs --force dp_policy --template-dir docs/templates
git subtree push --prefix docs/dp_policy origin gh-pages
```

### Running on server
To sync discrimination files:

```bash
rsync -avz results/policy_experiments/*.feather heinz:/home/rsteed/dp-policy/results/policy_experiments
```

To sync data files:
```bash
rsync -avz data/* heinz:/home/rsteed/dp-policy/data
```

To run R files, first [set the lib path](https://www.msi.umn.edu/support/faq/how-can-i-install-r-packages-my-home-directory).

Managing jobs:
```bash
scripts/kickoff.sh # or cp paste for jobs management
jobs
watch -n 60 ps
```

### Compressing for simple release
```bash
make zip
```

### Compressing PDFs for Overleaf
```bash
# NOTE: reduces DPI
# `/prepress` is highest DPI; then `/printer`; then `/ebook`
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -r100 -dPDFSETTINGS=/printer \
-dNOPAUSE -dQUIET -dBATCH -sOutputFile=[input]_compressed.pdf [input].pdf
# gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer \
# -dNOPAUSE -dQUIET -dBATCH -sOutputFile=[input]_compressed.pdf [input].pdf
```