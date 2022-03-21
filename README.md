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
rsync -avz results/policy_experiments/*.feather heinz:/home/rsteed/dp-acs/results/policy_experiments
```

To run R files, first [set the lib path](https://www.msi.umn.edu/support/faq/how-can-i-install-r-packages-my-home-directory).
