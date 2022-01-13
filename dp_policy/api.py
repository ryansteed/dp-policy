import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt

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
            grants["true_pop_total"],
            grants["true_children_total"],
            grants["true_children_poverty"],
            grants["cv"]
        )
    # get the noise-infused estimates - after sampling
    grants["dpest_pop_total"], \
        grants["dpest_children_total"], \
        grants["dpest_children_poverty"] = mechanism.poverty_estimates(
        grants["est_pop_total"],
        grants["est_children_total"],
        grants["est_children_poverty"]
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
    eps=list(np.logspace(-3, 10, num=10)) + [2.52], delta=[0.0],
    trials=1,
    mech_kwargs={},
    auth=False,
    allocator_kwargs={},
    verbose=False,
    print_results=[2.52, 0.1],
    plot_results=True
):
    allocations = []
    if verbose:
        print(f"{len(eps)*len(delta)*trials} iters:")
    for trial in tqdm(range(trials), desc='trial', disable=(not verbose)):
        for d in tqdm(delta, desc='delta', leave=False, disable=(not verbose)):
            for e in tqdm(eps, desc='eps', leave=False, disable=(not verbose)):
                allocations.append(titlei_funding(
                    SonnenbergAuthorizer,
                    saipe,
                    mech(e, d, **mech_kwargs),
                    get_sppe("../data/sppe18.xlsx"),
                    verbose=False,  # too noisy for a grid search
                    uncertainty=False,
                    allocator_kwargs=allocator_kwargs,
                    normalize=(not auth)
                ))
    results = pd.concat(
        allocations, axis=0,
        keys=itertools.product(range(trials), delta, eps),
        names=["trial", "delta", "epsilon"] + list(allocations[-1].index.names)
    )

    if print_results:
        eps, allocations = list(zip(*results.groupby("epsilon")))
        print(eps)

        for prefix in ("est", "dpest"):
            mse = []
            print("##", prefix)
            for e, alloc in results.groupby("epsilon"):
                for grant_type in (
                    "basic", "concentration", "targeted", "total"
                ):
                    error = alloc[f"{prefix}_grant_{grant_type}"] \
                        - alloc[f"true_grant_{grant_type}"]
                    if e in print_results:
                        print(f"## {grant_type} grants ##")
                        print(f"# districts: {len(error)}")
                        print("Average true alloc: {}".format(
                            alloc[f"true_grant_{grant_type}"].mean()
                        ))
                        print("Max true alloc: {}".format(
                            alloc[f"true_grant_{grant_type}"].max()
                        ))
                        print(f"Max error: {np.abs(error).max()}")
                        print(f"RMSE at eps={e}:", np.sqrt(np.mean(error**2)))
                        print(f"Total misalloc at eps={e}:", sum(abs(error)))
                        print(
                            "Total true alloc:",
                            sum(alloc[f"true_grant_{grant_type}"])
                        )

                    if grant_type == "total":
                        mse.append(np.sqrt(sum(error**2)/alloc.shape[0]))

            if plot_results:
                grant_type = "total"
                plt.plot(eps, mse)
                ax = plt.gca()
                ax.set_xscale('log')
                plt.xlabel("Epsilon")
                plt.ylabel(f"{grant_type} grant RMSE, nationally")
                plt.show()

                for i in range(len(eps)):
                    e = eps[i]
                    alloc = allocations[i][allocations[i][
                        "State Postal Code"] == "MI"
                    ]
                    alloc = alloc.sort_values(f"true_grant_{grant_type}")
                    plt.scatter(
                        range(len(alloc)),
                        alloc[f"{prefix}_grant_{grant_type}"] /
                        alloc[f"{prefix}_grant_{grant_type}"].sum(),
                        s=2, alpha=0.3, label=f"eps={e}"
                    )
                plt.scatter(
                    range(len(alloc)),
                    alloc[f"true_grant_{grant_type}"] /
                    sum(alloc[f"true_grant_{grant_type}"]),
                    s=2, alpha=0.3, label="true"
                )
                ax = plt.gca()
                ax.legend()
                ax.axes.xaxis.set_ticks([])
                ax.set_yscale('log')
                plt.xlabel("District (sorted by true alloc)")
                plt.ylabel("Allocation as % of total")
                plt.title(f"{grant_type} grants for Michigan")
                plt.show()

                for i in range(len(eps)):
                    e = eps[i]
                    alloc = allocations[i][allocations[i][
                        "State Postal Code"] == "MI"
                    ]
                    alloc['err_prop'] = (
                        alloc[f"{prefix}_grant_{grant_type}"] /
                        sum(alloc[f"{prefix}_grant_{grant_type}"]) -
                        alloc[f"true_grant_{grant_type}"] /
                        sum(alloc[f"true_grant_{grant_type}"])
                    ) * 1e6
                    plt.scatter(
                        alloc[f"true_grant_{grant_type}"] /
                        sum(alloc[f"true_grant_{grant_type}"]),
                        alloc.err_prop, s=3, alpha=0.4, label=f"eps={e}"
                    )
                ax = plt.gca()
                ax.legend()
                ax.set_xscale('log')
                ax.set_yscale('log')
                plt.xlabel("True allocation as % of total")
                plt.ylabel("Misallocation per million as % of total")
                plt.title(f"{grant_type} grants for Michigan")
                plt.show()

    return results
