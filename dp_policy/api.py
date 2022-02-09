import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt

from dp_policy.titlei.allocators import SonnenbergAuthorizer
from dp_policy.titlei.thresholders import PastThresholder
from dp_policy.titlei.utils import get_sppe, data


def titlei_funding(
    allocator, inputs, mechanism, sppe,
    normalize=True,
    allocator_kwargs={}, sampling_kwargs={},
    **grants_kwargs
):
    """
    Returns augmented SAIPE dataframe with randomized estimates and
    true/randomized grant amounts.
    """
    alloc = allocator(
        data(
            inputs, mechanism, sppe,
            sampling_kwargs=sampling_kwargs,
            **grants_kwargs
        ),
        **allocator_kwargs
    )
    return alloc.allocations(normalize=normalize)


def titlei_grid(
    inputs, mech,
    eps=list(np.logspace(-3, 10, num=10)) + [2.52], delta=[0.0],
    trials=1,
    mech_kwargs={},
    auth=False,
    allocator_kwargs={},
    verbose=True,
    print_results=[2.52, 0.1],
    plot_results=False
):
    allocations = []

    thresholder = allocator_kwargs.get('thresholder')
    if verbose:
        print(f"{len(eps)*len(delta)*trials} iters:")
    for trial in tqdm(range(trials), desc='trial', disable=(not verbose)):
        for d in tqdm(delta, desc='delta', leave=False, disable=True):
            for e in tqdm(eps, desc='eps', leave=False, disable=True):
                mechanism = mech(e, d, **mech_kwargs)
                sppe = get_sppe("../data/sppe18.xlsx")
                if thresholder is not None and isinstance(
                        thresholder, PastThresholder
                ):
                    thresholder.set_prior_estimates(
                        mechanism,
                        sppe,
                        verbose=False
                    )
                allocations.append(titlei_funding(
                    SonnenbergAuthorizer,
                    inputs,
                    mechanism,
                    sppe,
                    allocator_kwargs=allocator_kwargs,
                    verbose=False,  # too noisy for a grid search
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
                            "# small districts:",
                            len(alloc[alloc['true_pop_total'] < 20000])
                        )
                        print(
                            "Misalloc to large districts:",
                            np.abs(
                                error[alloc['true_pop_total'] >= 20000]
                            ).sum()
                        )
                        print(
                            "Misalloc to small districts:",
                            np.abs(
                                error[alloc['true_pop_total'] < 20000]
                            ).sum()
                        )
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
