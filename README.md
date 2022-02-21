# DP Policy Decision-Making

Author: Ryan Steed

WIP

```
make
```

## Running on server
To sync discrimination files:

```bash
rsync -avz results/policy_experiments/*.feather heinz:/home/rsteed/dp-acs/results/policy_experiments
```

To run R files, first [set the lib path](https://www.msi.umn.edu/support/faq/how-can-i-install-r-packages-my-home-directory).

## Onboarding (for Terrance)

Instructions for getting started with the bootstrap sampling distribution experiment.

The goal of this experiment is to randomly sample poverty estimates to get a sense for the distribution of allocations w/ and w/o DP.

A rough sketch of what I am thinking:

1.  For each district `i`, independently sample the poverty count `mu_i` from some prior distribution. For starters, you could do `mu_i ~ N(mu_i_hat, cv_i*mu_i_hat)` where `mu_i` is the true poverty count, `mu_i_hat` is the empirical poverty count given to us by the Census, and `cv_i` is the coefficient of variation also estimated by the Census. This is the distributional assumption the Census recommends for making confidence intervals - for more, read [this web page](https://www.census.gov/programs-surveys/saipe/guidance/district-estimates.html).

2. Add DP - simply sample some noise `e ~ Lap(2/eps)` from Laplace with sensitivity 2. I've been using `eps=0.1` for most experiments. Then compute `mu_i_dp = mu_i + e`.

3. Compute allocations given `[mu_1, mu_2, ..., mu_n]` and allocations given `[mu_1_dp, mu_2_dp, ..., mu_n_dp]`.

4. Run 1-3 many times and compare the distributions.

Code-wise,

- [Step 1] For constructing the priors, you'll want the empirical poverty counts table, which you can generate with either `dp_policy.titlei.utils.get_saipe` (SAIPE = "Small Area Income and Poverty Estimates") or `dp_policy.api.titlei_data`. I would use the latter, but ignore the columns prefixed with "est" - those are precomputed DP estimates. For examples of how to use these methods, see `notebooks/titlei.ipynb`. The relevant columns are `true_children_eligible` (this is `mu_hat`) and `cv` (this is `cv`).

- [Step 2] `dp_policy.titlei.mechanisms` wraps a library for DP mechanisms I found - this is how I'm quickly adding the DP noise. Could be easily adapted for step 2, or you could just draw your own samples.

- [Step 3] `dp_policy.titlei.allocators.SonnenbergAuthorizer`, the main allocation calculator - for step 3, turning your table of estimates into allocations. I would just copy the method invocations and parameters used in `dp_policy.api.titlei_funding` and `notebooks/titlei.ipynb`

For visualizations and other analysis, there may already be useful code in `notebooks/titlei.ipynb`, depending on what you want to do.
