import numpy as np
import pandas as pd

from dp_policy.api import titlei_funding as funding
from dp_policy.titlei.mechanisms import Mechanism, GroundTruth
from dp_policy.titlei.allocators import SonnenbergAuthorizer

from tqdm.notebook import tqdm

COLS_INDEX = ['State FIPS Code', 'District ID']
COLS_GROUPBY = ['State FIPS Code', 'District ID', 'State Postal Code', 'Name']
COLS_GRANT = ['basic', 'concentration', 'targeted']


class DummyMechanism():
    def randomise(self, x):
        return x


class Sampled(Mechanism):
    def __init__(self, saipe, round=False, clip=True,
                 mechanism=None, epsilon=0.1, delta=1e-6, sensitivity=2.0,
                 **kwargs):
        super().__init__(epsilon, delta, sensitivity, **kwargs)
        self.saipe = saipe
        self.round = round
        self.clip = clip

        if mechanism is None:
            self.mechanism = DummyMechanism()
        else:
            self.mechanism = mechanism(
                epsilon=self.epsilon,
                delta=self.delta,
                sensitivity=self.sensitivity
            )

    def post_processing(self, count):
        if self.round:
            count = np.round(count)
        if self.clip:
            count = np.clip(count, 0, None)
        return count

    def poverty_estimates(self):
        saipe = self.saipe.copy()

        pop_total = \
            saipe["Estimated Total Population"].apply(self.mechanism.randomise)
        children_total = \
            saipe["Estimated Population 5-17"].apply(self.mechanism.randomise)

        children_poverty = saipe[
            "Estimated number of relevant children 5 to 17 years old"
            " in poverty who are related to the householder"
        ]
        mu = np.clip(children_poverty.values, 0, None)
        cv = saipe['cv'].values
        children_poverty.loc[:] = np.random.normal(mu, mu * cv)
        children_poverty = children_poverty.apply(self.mechanism.randomise)

        return self.post_processing(pop_total), \
            self.post_processing(children_total), \
            self.post_processing(children_poverty)


def collect_results(
    saipe, mech, sppe, num_runs=1,
    quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
):
    cols_keep = ["true_grant_{}".format(col) for col in COLS_GRANT] + \
                ["est_grant_{}".format(col) for col in COLS_GRANT] + \
                ["true_eligible_{}".format(col) for col in COLS_GRANT] + \
                ["est_eligible_{}".format(col) for col in COLS_GRANT]
    results = []
    for i in tqdm(range(num_runs)):
        allocations = funding(
            SonnenbergAuthorizer, saipe, mech, sppe, verbose=False
        )
        allocations = allocations.reset_index().set_index(COLS_GROUPBY)
        allocations = allocations[cols_keep]
        allocations['run'] = i
        results.append(allocations)
    results = pd.concat(results)
    for col in COLS_GRANT:
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
