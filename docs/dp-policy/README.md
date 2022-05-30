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
dp_policy run [name]
# to only produce the feather file for regression analysis (using cached results)
dp_policy run --just-join [name]
# to run all experiments
dp_policy run_all
```

## Running on server
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

## Compressing for simple release
```bash
make zip
```

# Documentation
Documentation for the `dp-policy` API is published at [rbsteed.com/dp-policy](https://rbsteed.com/dp-policy).

To generate the documentation, use pdoc3:

```bash
pdoc3 --html --output-dir docs --force dp_policy --template-dir docs/templates
git subtree push --prefix docs/dp-policy origin gh-pages
```
