import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools

from dp_policy.titlei.allocators import SonnenbergAuthorizer
from dp_policy.titlei.utils import get_sppe
from dp_policy.titlei.mechanisms import Sampled


def titlei_data(
    saipe, mechanism, sppe, verbose=True
):
    # ground truth - assume SAIPE 2019 is ground truth
    grants = saipe.rename(columns={
        "Estimated Total Population": "true_pop_total",
        "Estimated Population 5-17": "true_children_total",
        "Estimated number of relevant children 5 to 17 years old in poverty"
        " who are related to the householder": "true_children_poverty"
    })
    # sample from the sampling distribution
    mechanism_sampling = Sampled()
    grants["est_pop_total"], \
        grants["est_children_total"], \
        grants["est_children_poverty"] = mechanism_sampling.poverty_estimates(
            grants["true_pop_total"].values,
            grants["true_children_total"].values,
            grants["true_children_poverty"].values,
            grants["cv"].values
        )
    # get the noise-infused estimates - after sampling
    grants["dpest_pop_total"], \
        grants["dpest_children_total"], \
        grants["dpest_children_poverty"] = mechanism.poverty_estimates(
        grants["est_pop_total"].values,
        grants["est_children_total"].values,
        grants["est_children_poverty"].values
    )
    # back out the noise-infused estimates - before sampling
    # doing it this way because we want to see the same noise draws added to
    # both bases - not a separate draw here
    for var in ("pop_total", "children_total", "children_poverty"):
        grants[f"dp_{var}"] = \
            grants[f"dpest_{var}"] - grants[f"est_{var}"] \
            + grants[f"true_{var}"]

    # BIG ASSUMPTION, TODO: revisit later
    for prefix in ("true", "est", "dp", "dpest"):
        grants[f"{prefix}_children_eligible"] = grants[
            f"{prefix}_children_poverty"
        ]

    # join in SPPE
    grants = grants.reset_index()\
        .merge(sppe, left_on="State Postal Code", right_on="abbrv")\
        .drop(columns=['abbrv', 'state']).rename(columns={'ppe': 'sppe'})\
        .set_index(["State FIPS Code", "District ID"])

    if verbose:
        print(
            "[WARN] Dropping districts with missing SPPE data:",
            grants[grants.sppe.isna()]['Name'].values
        )
    grants = grants.dropna(subset=["sppe"])
    grants.sppe = grants.sppe.astype(float)

    return grants


def titlei_funding(
    allocator, saipe, mechanism, sppe,
    uncertainty=False, normalize=True, allocator_kwargs={},
    **grants_kwargs
):
    """
    Returns augmented SAIPE dataframe with randomized estimates and
    true/randomized grant amounts.
    """
    alloc = allocator(
        titlei_data(saipe, mechanism, sppe, **grants_kwargs),
        **allocator_kwargs
    )
    return alloc.allocations(uncertainty=uncertainty, normalize=normalize)


def titlei_grid(
    saipe, mech,
    eps=list(np.logspace(-3, 1)) + [2.5], delta=[0.0],
    trials=1,
    mech_kwargs={},
    auth=False
):
    allocations = []
    print(f"{len(eps)*len(delta)*trials} iters:")
    for trial in tqdm(range(trials), desc='trial'):
        for d in tqdm(delta, desc='delta', leave=False):
            for e in tqdm(eps, desc='eps', leave=False):
                allocations.append(titlei_funding(
                    SonnenbergAuthorizer,
                    saipe,
                    mech(e, d, **mech_kwargs),
                    get_sppe("../data/sppe18.xlsx"),
                    verbose=False,
                    uncertainty=False,
                    normalize=(not auth)
                ))
    return pd.concat(
        allocations, axis=0,
        keys=itertools.product(range(trials), delta, eps),
        names=["trial", "delta", "epsilon"] + list(allocations[-1].index.names)
    )
