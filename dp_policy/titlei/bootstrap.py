import pandas as pd
from tqdm.notebook import tqdm

from dp_policy.api import titlei_funding as funding
from dp_policy.titlei.allocators import SonnenbergAuthorizer

COLS_INDEX = ['State FIPS Code', 'District ID']
COLS_GROUPBY = ['State FIPS Code', 'District ID', 'State Postal Code', 'Name']
COLS_GRANT = ['basic', 'concentration', 'targeted', 'total']


def collect_results(
    saipe, mech, sppe, num_runs=1,
    quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
):
    cols_keep = ["true_grant_{}".format(col) for col in COLS_GRANT] + \
                ["est_grant_{}".format(col) for col in COLS_GRANT] + \
                [
                    "true_eligible_{}".format(col)
                    for col in COLS_GRANT if col != 'total'
                ] + \
                [
                    "est_eligible_{}".format(col)
                    for col in COLS_GRANT if col != 'total'
                ]
    results = []
    for i in tqdm(range(num_runs)):
        allocations = funding(
            SonnenbergAuthorizer, saipe, mech, sppe,
            verbose=False, uncertainty=False, normalize=True
        )
        allocations = allocations.reset_index().set_index(COLS_GROUPBY)
        allocations = allocations[cols_keep]
        allocations['run'] = i
        results.append(allocations)
    results = pd.concat(results)
    for col in [c for c in COLS_GRANT if c != 'total']:
        results["diff_grant_{}".format(col)] = (
            results["est_grant_{}".format(col)].astype(float) -
            results["true_grant_{}".format(col)].astype(float)
        )
        results["diff_eligible_{}".format(col)] = (
            results["est_eligible_{}".format(col)].astype(float) -
            results["true_eligible_{}".format(col)].astype(float)
        )
        results["diff_eligible_{}".format(col)] = \
            (results["diff_eligible_{}".format(col)] < 0).astype(float)
    x = results.abs().groupby('run')

    results_dict = {}
    results_dict['sum'] = x.sum()
    results_dict['mean'] = x.mean()
    for quantile in quantiles:
        results_dict[quantile] = x.quantile(quantile)
    return results, results_dict
