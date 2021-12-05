import pickle
import numpy as np
import pandas as pd

from diffprivlib.mechanisms import Laplace as LaplaceMech
from diffprivlib.mechanisms import GaussianAnalytic as GaussianMech

from dp_policy.api import titlei_funding as funding
from dp_policy.titlei.mechanisms import Mechanism, GroundTruth
from dp_policy.titlei.utils import get_saipe, get_sppe
from dp_policy.titlei.allocators import SonnenbergAuthorizer

import pdb

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

        pop_total = saipe["Estimated Total Population"].apply(self.mechanism.randomise)
        children_total = saipe["Estimated Population 5-17"].apply(self.mechanism.randomise)

        children_poverty = saipe["Estimated number of relevant children 5 to 17 years old"
                                      " in poverty who are related to the householder"]
        mu = np.clip(children_poverty.values, 0, None)
        cv = saipe['cv'].values
        children_poverty.loc[:] = np.random.normal(mu, (mu * cv) ** 0.5)
        children_poverty = children_poverty.apply(self.mechanism.randomise)

        return self.post_processing(pop_total), \
               self.post_processing(children_total), \
               self.post_processing(children_poverty)

# def collect_results(saipe, mech, sppe, num_runs=1, quantiles=()):
#     cols_index = ['State FIPS Code', 'District ID']
#     cols_groupby = ['State FIPS Code', 'District ID', 'State Postal Code', 'Name']
#
#     results = []
#     for i in range(num_runs):
#         allocations = funding(SonnenbergAuthorizer, saipe, mech, sppe)
#         allocations['run'] = i
#         pdb.set_trace()
#         results.append(allocations)
#
#     results = pd.concat(results).reset_index()
#     # results = results.groupby(cols_groupby)
#     for col in cols_groupby:
#         del results[col]
#     results = results.groupby('run')
#
#     pdb.set_trace()
#
#     results_dict = {}
#     results_dict['mean'] = results.mean()
#     for quantile in quantiles:
#         results_dict[quantile] = results.quantile(quantile)\
#
#     for key, val in results_dict.items():
#         results_dict[key] = results_dict[key].reset_index().set_index(cols_index)
#
#     return results_dict

def collect_results(saipe, mech, sppe, num_runs=1, quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    cols_keep = ["true_grant_{}".format(col) for col in COLS_GRANT] + \
                ["est_grant_{}".format(col) for col in COLS_GRANT] + \
                ["true_eligible_{}".format(col) for col in COLS_GRANT] + \
                ["est_eligible_{}".format(col) for col in COLS_GRANT]
    results = []
    for i in range(num_runs):
        allocations = funding(SonnenbergAuthorizer, saipe, mech, sppe)
        allocations = allocations.reset_index().set_index(COLS_GROUPBY)
        allocations = allocations[cols_keep]
        allocations['run'] = i
        results.append(allocations)
    results = pd.concat(results)
    for col in COLS_GRANT:
        results["diff_grant_{}".format(col)] = results["est_grant_{}".format(col)].astype(float) - results["true_grant_{}".format(col)].astype(float)
        results["diff_eligible_{}".format(col)] = results["est_eligible_{}".format(col)].astype(float) - results["true_eligible_{}".format(col)].astype(float)
        results["diff_eligible_{}".format(col)] = (results["diff_eligible_{}".format(col)] < 0).astype(float)
    x = results.abs().groupby('run')

    results_dict = {}
    results_dict['sum'] = x.sum()
    results_dict['mean'] = x.mean()
    for quantile in quantiles:
        results_dict[quantile] = x.quantile(quantile)
    return results_dict

NUM_RUNS = 100

saipe = get_saipe("../data/saipe19.xls")
sppe = get_sppe("../data/sppe18.xlsx")

# nonprivate
mech = Sampled(saipe)
results_nondp = collect_results(saipe, mech, sppe, num_runs=NUM_RUNS)
with open('sampling_results/results_nondp.pkl', 'wb') as handle:
    pickle.dump(results_nondp, handle)

delta = 1e-6
for epsilon in [0.1, 0.5, 1.0]:
    # Laplace
    mech = Sampled(saipe, mechanism=LaplaceMech, epsilon=epsilon, delta=delta)
    results_laplace = collect_results(saipe, mech, sppe, num_runs=NUM_RUNS)
    with open('sampling_results/results_laplace_eps{}.pkl'.format(epsilon), 'wb') as handle:
        pickle.dump(results_laplace, handle)

    # Gaussian
    mech = Sampled(saipe, mechanism=GaussianMech, epsilon=epsilon, delta=delta)
    results_gaussian = collect_results(saipe, mech, sppe, num_runs=NUM_RUNS)
    with open('sampling_results/results_gaussian_eps{}.pkl'.format(epsilon), 'wb') as handle:
        pickle.dump(results_gaussian, handle)