import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt

import dp_policy.config as config
from dp_policy.titlei.evaluation import \
    discrimination_treatments_join, save_treatments, load_treatments, \
    match_true, compare_treatments
from dp_policy.titlei.mechanisms import Laplace
from dp_policy.titlei.utils import get_inputs, data, get_sppe
from dp_policy.titlei.allocators import SonnenbergAuthorizer
from dp_policy.titlei.thresholders import \
    AverageThresholder, RepeatThresholder, DummyThresholder, MOEThresholder, \
    PastThresholder


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
    trials=100,
    mech_kwargs={},
    auth=False,
    allocator_kwargs={},
    sampling_kwargs={},
    verbose=True,
    print_results=[2.52, 0.1],
    plot_results=False,
    alpha=0.05,
    results=None
):
    if results is None:
        allocations = []
        thresholder = allocator_kwargs.get('thresholder')
        if verbose:
            print(f"{len(eps)*len(delta)*trials} iters:")
        for trial in tqdm(range(trials), desc='trial', disable=(not verbose)):
            for d in tqdm(delta, desc='delta', leave=False, disable=True):
                for e in tqdm(eps, desc='eps', leave=False, disable=True):
                    mechanism = mech(e, d, **mech_kwargs)
                    sppe = get_sppe(os.path.join(
                        config.root,
                        "data/sppe18.xlsx"
                    ))
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
                        sampling_kwargs=sampling_kwargs,
                        verbose=False,  # too noisy for a grid search
                        normalize=(not auth)
                    ))
        results = pd.concat(
            allocations, axis=0,
            keys=itertools.product(range(trials), delta, eps),
            names=[
                "trial", "delta", "epsilon"
            ] + list(allocations[-1].index.names)
        )

    if print_results:
        eps, allocations = list(zip(*results.groupby("epsilon")))

        for prefix in ("est", "dp", "dpest"):
            print("##", prefix)
            for e, alloc in results.groupby("epsilon"):
                for grant_type in (
                    "basic", "concentration", "targeted", "total"
                ):
                    error = alloc[f"{prefix}_grant_{grant_type}"] \
                        - alloc[f"true_grant_{grant_type}"]
                    err_grouped = error.groupby(
                        ["State FIPS Code", "District ID"]
                    )
                    exp_error = err_grouped.mean()
                    if e in print_results:
                        print(f"## {grant_type} grants - eps={e} ##")
                        print(f"# districts: {len(error)}")
                        print("Average true alloc: {}".format(
                            alloc[f"true_grant_{grant_type}"].mean()
                        ))
                        print("Max true alloc: {}".format(
                            alloc[f"true_grant_{grant_type}"].max()
                        ))
                        print(f"Max error: {np.abs(error).max()}")
                        print(f"RMSE:", np.sqrt(np.mean(error**2)))
                        print(
                            f"RMSE in exp. error:",
                            np.sqrt(np.mean(exp_error**2))
                        )
                        print(
                            f"Avg. sum of negative misallocs:",
                            np.sum(np.abs(exp_error[exp_error < 0]))
                        )
                        print(f"Avg. total misalloc:", sum(abs(exp_error)))

                        small_district = alloc["true_pop_total"]\
                            .groupby(["State FIPS Code", "District ID"])\
                            .first() < 20000
                        print(
                            "# small districts:",
                            small_district.sum()
                        )
                        print(
                            "Total avg misalloc to large districts:",
                            exp_error[~small_district].abs().sum()
                        )
                        print(
                            "Total avg misalloc to small districts:",
                            exp_error[small_district].abs().sum()
                        )
                        print(
                            "Avg total true alloc:",
                            alloc[f"true_grant_{grant_type}"]
                            .groupby(["State FIPS Code", "District ID"])
                            .first().abs().sum()
                        )

            if plot_results:
                trials = len(results.groupby("trial"))

                grant_type = "total"

                error = results[f"{prefix}_grant_{grant_type}"] \
                    - results[f"true_grant_{grant_type}"]
                print(error.shape)
                rmse = np.sqrt(error.pow(2).groupby(
                    ["epsilon", "trial"]
                ).mean())
                print(rmse)
                mse = {
                    'mean': rmse.groupby("epsilon").mean(),
                    'lower': rmse.groupby("epsilon").quantile(alpha/2),
                    'upper': rmse.groupby("epsilon").quantile(1-alpha/2)
                }
                plt.plot(eps, mse['mean'])
                plt.fill_between(
                    eps, mse['lower'], mse['upper'],
                    color='gray', alpha=0.25
                )
                ax = plt.gca()
                ax.set_xscale('log')
                plt.xlabel("Epsilon")
                plt.ylabel(
                    f"Avg. RMSE in {grant_type} grants over {trials} trials"
                )
                plt.savefig(
                    os.path.join(
                        config.root,
                        "plots/robustness/eps_sensitivity_frontier.png"
                    ),
                    dpi=300
                )
                plt.show()

                # for i in range(len(eps)):
                #     e = eps[i]
                #     alloc = allocations[i][allocations[i][
                #         "State Postal Code"] == "MI"
                #     ]
                #     alloc = alloc.sort_values(f"true_grant_{grant_type}")
                #     plt.scatter(
                #         range(len(alloc)),
                #         alloc[f"{prefix}_grant_{grant_type}"] /
                #         alloc[f"{prefix}_grant_{grant_type}"].sum(),
                #         s=2, alpha=0.3, label=f"eps={e}"
                #     )
                # plt.scatter(
                #     range(len(alloc)),
                #     alloc[f"true_grant_{grant_type}"] /
                #     sum(alloc[f"true_grant_{grant_type}"]),
                #     s=2, alpha=0.3, label="true"
                # )
                # ax = plt.gca()
                # ax.legend()
                # ax.axes.xaxis.set_ticks([])
                # ax.set_yscale('log')
                # plt.xlabel("District (sorted by true alloc)")
                # plt.ylabel("Allocation as % of total")
                # plt.title(f"{grant_type} grants for Michigan")
                # plt.show()

                # for i in range(len(eps)):
                #     e = eps[i]
                #     alloc = allocations[i][allocations[i][
                #         "State Postal Code"] == "MI"
                #     ]
                #     alloc['err_prop'] = (
                #         alloc[f"{prefix}_grant_{grant_type}"] /
                #         sum(alloc[f"{prefix}_grant_{grant_type}"]) -
                #         alloc[f"true_grant_{grant_type}"] /
                #         sum(alloc[f"true_grant_{grant_type}"])
                #     ) * 1e6
                #     plt.scatter(
                #         alloc[f"true_grant_{grant_type}"] /
                #         sum(alloc[f"true_grant_{grant_type}"]),
                #         alloc.err_prop, s=3, alpha=0.4, label=f"eps={e}"
                #     )
                # ax = plt.gca()
                # ax.legend()
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # plt.xlabel("True allocation as % of total")
                # plt.ylabel("Misallocation per million as % of total")
                # plt.title(f"{grant_type} grants for Michigan")
                # plt.show()

    return results


class Experiment:
    def __init__(
        self,
        name,
        baseline="cached",
        year=2021,
        trials=1000,
        eps=[0.1],
        delta=[0.0]
    ):
        self.name = name
        self.trials = trials
        self.eps = eps
        self.delta = delta
        self.saipe = get_inputs(year)

        if str(baseline) == "cached":
            print("Using cached baseline...")
            try:
                self.baseline = load_treatments("baseline")['baseline']
            except FileNotFoundError:
                print("[WARN] Could not find cached baseline, generating.")
                self._generate_baseline()
        elif baseline is None:
            print("Generating baseline...")
            self._generate_baseline()
        else:
            print("Using given baseline...")
            self.baseline = baseline

    def _generate_baseline(self):
        self.baseline = titlei_grid(
            self.saipe, Laplace,
            eps=self.eps, delta=self.delta,
            trials=self.trials,
            print_results=False,
            allocator_kwargs={'verbose': False}
        )
        save_treatments({'baseline': self.baseline}, "baseline")
        discrimination_treatments_join("baseline")

    def run(self):
        treatments = self._get_treatments()
        save_treatments(treatments, self.name)
        self.discrimination_join()

    def _get_treatments(self):
        raise NotImplementedError

    def discrimination_join(self, **join_kwargs):
        discrimination_treatments_join(self.name, **join_kwargs)

    def plot(self, **kwargs):
        treatments = load_treatments(self.name)
        compare_treatments(treatments, experiment_name=self.name, **kwargs)

    @staticmethod
    def get_experiment(name, *args, **kwargs):
        experiments = {
            'baseline': Baseline,
            'hold_harmless': HoldHarmless,
            'post_processing': PostProcessing,
            'thresholds': Thresholds,
            'epsilon': Epsilon,
            'moving_average': MovingAverage,
            'budget': Budget,
            'sampling': Sampling
        }
        Exp = experiments.get(name)
        if Exp is None:
            raise ValueError(f"{name} not a supported experiment name.")
        return Exp(name, *args, **kwargs)


class Baseline(Experiment):
    def _get_treatments(self):
        return {
            'baseline': self.baseline
        }


class HoldHarmless(Experiment):
    def _get_treatments(self):
        hold_harmless = titlei_grid(
            self.saipe, Laplace,
            eps=self.eps, delta=self.delta,
            trials=self.trials, print_results=False,
            allocator_kwargs={'hold_harmless': True}
        )
        # ground truths are the same
        match_true(self.baseline, [hold_harmless])

        # save treatments to file for later
        return {
            'No hold harmless (baseline)': self.baseline,
            'Hold harmless': hold_harmless
        }


class PostProcessing(Experiment):
    def _get_treatments(self):
        none = titlei_grid(
            self.saipe, Laplace,
            eps=self.eps,
            delta=self.delta,
            trials=self.trials,
            mech_kwargs=dict(
                clip=False,
                round=False
            ),
            print_results=False
        )
        clipping = self.baseline
        rounding = titlei_grid(
            self.saipe, Laplace,
            eps=self.eps,
            delta=self.delta,
            trials=self.trials,
            mech_kwargs=dict(
                clip=True,
                round=True
            ),
            print_results=False
        )
        match_true(self.baseline, [none, rounding])
        return {
            'None': none,
            'Clipping (baseline)': clipping,
            'Clipping + Rounding': rounding
        }


class MovingAverage(Experiment):
    def __init__(self, name, truth="average", **kwargs):
        if truth != "average":
            raise Exception(
                "Not supporting truth values other than `average`."
            )
        self.truth = truth
        super().__init__(f"{name}_truth={truth}", **kwargs)

    def _get_treatments(self):
        single_year = self.baseline
        averaged = [
            titlei_grid(
                get_inputs(2021, avg_lag=i+1),
                Laplace,
                eps=self.eps,
                delta=self.delta,
                trials=self.trials,
                print_results=False
            )
            for i in range(4)
        ]
        match_true(averaged[-1], [single_year] + averaged[:-1])
        return {
            'Lag 0': single_year,
            **{f"Lag {i+1}": a for i, a in enumerate(averaged)}
        }


class Thresholds(Experiment):
    def _get_treatments(self):
        # hard threshold
        hard = self.baseline

        # use average
        averaged = titlei_grid(
            self.saipe,
            Laplace,
            eps=self.eps,
            delta=self.delta,
            trials=self.trials,
            allocator_kwargs={
                'thresholder': AverageThresholder(2021, 4)
            },
            print_results=False
        )

        # must be ineligible 2x in a row
        repeat2 = titlei_grid(
            self.saipe,
            Laplace,
            eps=self.eps,
            delta=self.delta,
            trials=self.trials,
            allocator_kwargs={
                'thresholder': RepeatThresholder(2021, 1)
            },
            print_results=False
        )

        # must be ineligible 3x in a row
        repeat3 = titlei_grid(
            self.saipe,
            Laplace,
            eps=self.eps,
            delta=self.delta,
            trials=self.trials,
            allocator_kwargs={
                'thresholder': RepeatThresholder(2021, 2)
            },
            print_results=False
        )

        # no threshold
        none = titlei_grid(
            self.saipe,
            Laplace,
            eps=self.eps,
            delta=self.delta,
            trials=self.trials,
            allocator_kwargs={
                'thresholder': DummyThresholder()
            },
            print_results=False
        )

        # moe
        moe_01 = titlei_grid(
            self.saipe,
            Laplace,
            eps=self.eps,
            delta=self.delta,
            trials=self.trials,
            allocator_kwargs={
                'thresholder': MOEThresholder(alpha=0.1)
            },
            print_results=False
        )

        match_true(hard, [averaged, repeat2, repeat3, none, moe_01])
        return {
            'None': none,
            'Hard (baseline)': hard,
            'Averaged': averaged,
            'Repeated years (2)': repeat2,
            'Repeated years (3)': repeat3,
            'Margin of error (90%)': moe_01
        }


class Budget(Experiment):
    def _get_treatments(self):
        if len(self.eps) > 0:
            print("[WARN] Using first value of epsilon for budget calcs.")

        e = self.eps[0]
        alpha = 0.05
        test = self.baseline.loc[pd.IndexSlice[:, 0.0, e, :, :], :]

        budgets = {
            "Biden proposal": 2e10
        }
        for name, prefixes in {
            "+ loss": ("dpest", "true"),
            # "+ marginal loss (DP)": ("dpest", "est"),
            # "+ loss (data error)": ("est", "true")
        }.items():
            error = test[f"{prefixes[1]}_grant_total"] \
                - test[f"{prefixes[0]}_grant_total"]
            err_grouped = error.groupby(
                ["State FIPS Code", "District ID"]
            )
            exp_error = err_grouped.mean()

            budgets[name] = exp_error[exp_error < 0].abs().sum()

            if prefixes == ("dpest", "true"):
                # how much money to remove alpha quantile loss?
                quantile = err_grouped.quantile(alpha)
                budgets[f"+ {(alpha)*100}% quant. loss"] = \
                    quantile[quantile < 0].abs().sum()

                # how much money to cover all misallocated dollars?
                # budgets[f"+ exp. misalloc."] = exp_error.abs().sum()

        budgets = {"{} (${:.1e})".format(k, v): v for k, v in budgets.items()}

        usual_budget = self.saipe["official_total_alloc"]\
            .groupby(["State FIPS Code", "District ID"]).first().sum()
        print("baseline budget: {:.1e}".format(usual_budget))
        print(budgets)

        treatments = {
            name: titlei_grid(
                self.saipe, Laplace,
                eps=[e], delta=self.delta,
                trials=self.trials, print_results=False,
                allocator_kwargs={
                    'appropriation': round(budget + usual_budget, 2)
                }
            ) for name, budget in budgets.items()
        }
        match_true(self.baseline, list(treatments.values()))
        treatments["FY2019 appropriation (baseline)"] = test

        return treatments


class Epsilon(Experiment):
    def _get_treatments(self):
        return {
            e: titlei_grid(
                self.saipe, Laplace,
                eps=[e], delta=self.delta,
                trials=self.trials, print_results=False
            ) if e not in self.eps
            else self.baseline.loc[pd.IndexSlice[:, 0.0, e, :, :], :].copy()
            for e in [1e-3, 1e-2, 1e-1, 1, 10, 30]
        }

    def discrimination_join(self):
        return super().discrimination_join(epsilon=None)

    def plot(self):
        return super().plot(epsilon=None)


class Sampling(Experiment):
    def _get_treatments(self):
        gaussian = {
            f"Gaussian ({m})": titlei_grid(
                self.saipe, Laplace,
                eps=self.eps, delta=self.delta,
                trials=self.trials, print_results=False,
                sampling_kwargs=dict(
                    multiplier=m,
                    distribution="gaussian"
                )
            )
            for m in [0.5, 0.75, 1, 1.5]
        }
        laplace = {
            f"Laplace ({m})": titlei_grid(
                self.saipe, Laplace,
                eps=self.eps, delta=self.delta,
                trials=self.trials, print_results=False,
                sampling_kwargs=dict(
                    multiplier=m,
                    distribution="laplace"
                )
            )
            for m in [0.5, 1, 1.5]
        }
        return {
            **gaussian,
            **laplace
        }
