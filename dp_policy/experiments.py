import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt

import dp_policy.config as config
from dp_policy.titlei.evaluation import \
    discrimination_treatments_join, save_treatments, load_treatments, \
    match_true, compare_treatments, misalloc_statistics
from dp_policy.titlei.mechanisms import Mechanism, Laplace
from dp_policy.titlei.utils import get_inputs, data, get_sppe
from dp_policy.titlei.allocators import Allocator, SonnenbergAuthorizer
from dp_policy.titlei.thresholders import \
    AverageThresholder, RepeatThresholder, DummyThresholder, MOEThresholder, \
    PastThresholder

from typing import List, Union


def titlei_funding(
    allocator: Allocator,
    inputs: pd.DataFrame,
    mechanism: Mechanism,
    sppe: pd.DataFrame,
    normalize: bool = True,
    allocator_kwargs: dict = {},
    sampling_kwargs: dict = {},
    **grants_kwargs
):
    """Allocate Title I funding.

    Args:
        allocator (Allocator): The allocator to use.
        inputs (pd.DataFrame): The necessary inputs for the allocator -
            usually the SAIPE - by district.
        mechanism (Mechanism): The randomization mechanism to use.
        sppe (pd.DataFrame): State per-pupil education data.
        normalize (bool, optional): Whether to normalize the authorization
            amounts. Defaults to True.
        allocator_kwargs (dict, optional): Parameters for the allocator.
            Defaults to {}.
        sampling_kwargs (dict, optional): Parameters for the sampling
            mechanism. Defaults to {}.

    Returns:
        pd.DataFrame: Returns input dataframe with grant allocations under
            ground truth, sampling, and privacy randomization.
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
    inputs: pd.DataFrame,
    mech: Mechanism,
    eps: list = list(np.logspace(-3, 10, num=10)) + [2.52],
    delta: list = [0.0],
    trials: int = 100,
    mech_kwargs: dict = {},
    auth: bool = False,
    allocator_kwargs: dict = {},
    sampling_kwargs: dict = {},
    verbose: bool = True,
    print_results: list = [2.52, 0.1],
    plot_results: bool = False,
    alpha: float = 0.05,
    results: pd.DataFrame = None
):
    """Run many trials of Title I process under different privacy parameters.

    Args:
        inputs (pd.DataFrame): The necessary inputs for the allocator -
            usually the SAIPE - by district.
        mech (Mechanism): The randomization mechanism to use.
        eps (list, optional): Values of epsilon to try. Defaults to
            `list(np.logspace(-3, 10, num=10))+[2.52]`.
        delta (list, optional): Values of delta to try. Defaults to [0.0].
        trials (int, optional): Number of trials to run for each parameter
            combo. Defaults to 100.
        mech_kwargs (dict, optional): Parameters for the randomization
            mechanism. Defaults to {}.
        auth (bool, optional): Whether to use the raw authorization amounts.
            Defaults to False.
        allocator_kwargs (dict, optional): Parameters for the allocator.
            Defaults to {}.
        sampling_kwargs (dict, optional): Parameters for the sampling
            mechanism. Defaults to {}.
        verbose (bool, optional): Defaults to True.
        print_results (list, optional): Which values of epsilon to print
            results for. Defaults to [2.52, 0.1].
        plot_results (bool, optional): Whether to plot results. Defaults to
            False.
        alpha (float, optional): Confidence level for confidence bands.
            Defaults to 0.05.
        results (pd.DataFrame, optional): Pre-computed results to plot. If
            provided, will skip grid calculation. Defaults to None.

    Returns:
        pd.DataFrame: Results grid.
    """
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
        prefixes = ["est", "dp", "dpest"]

        for e, alloc in results.groupby("epsilon"):
            if e in print_results:
                print(f"--- eps={e} ---")
                data_error = alloc["est_children_eligible"] \
                    - alloc["true_children_eligible"]
                dp_error = alloc["dpest_children_eligible"] \
                    - alloc["est_children_eligible"]
                s = 0.5
                plt.scatter(
                    alloc["true_children_eligible"], data_error,
                    s, label="data"
                )
                plt.scatter(
                    alloc["true_children_eligible"], dp_error,
                    s, label="dp"
                )
                plt.legend()
                plt.xlabel("# children in poverty")
                plt.ylabel("Noise")
                plt.show()

                plt.scatter(
                    alloc["true_children_eligible"],
                    data_error/alloc["true_children_eligible"],
                    s, label="data"
                )
                plt.scatter(
                    alloc["true_children_eligible"],
                    dp_error/alloc["true_children_eligible"],
                    s, label="dp"
                )
                plt.legend()
                plt.xlabel("# children in poverty")
                plt.ylabel("Noise per child in poverty")
                plt.show()

        for prefix in prefixes:
            print("##", prefix)
            for e, alloc in results.groupby("epsilon"):
                for grant_type in (
                    "basic", "concentration", "targeted", "total"
                ):
                    if e in print_results:
                        error = alloc[f"{prefix}_grant_{grant_type}"] \
                            - alloc[f"true_grant_{grant_type}"]
                        print(f"## {grant_type} grants - eps={e} ##")
                        misalloc_statistics(
                            error,
                            allocations=alloc,
                            grant_type=grant_type
                        )

            if plot_results:
                trials = len(np.unique(
                    results.index.get_level_values("trial")
                ))

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
                eps = mse['mean'].index
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

    return results


class Experiment:
    """Handles running and setting experiments.
    """
    def __init__(
        self,
        name: str,
        baseline: Union[str, pd.DataFrame] = "cached",
        year: int = 2021,
        trials: int = 1000,
        eps: List[float] = [0.1],
        delta: List[float] = [0.0]
    ):
        """Initialize experiment and set baseline.

        Args:
            name (str): Experiment name.
            baseline (Union[str, pd.DataFrame], optional): What baseline
                (control) results to use. Defaults to "cached".
            year (int, optional): What year to get inputs for. Defaults to
                2021.
            trials (int, optional): Number of trials to run for each condition.
                Defaults to 1000.
            eps (List[float], optional): Values of epsilon to try. Defaults to
                [0.1].
            delta (List[float], optional): Values of delta to try. Defaults to
                [0.0].
        """
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
        """Generate the control condition. Defaults to normal Laplace
            mechanism with no special allocation parameters.
        """
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
        """Run the experiment, save results, and create joined covariate file.
        """
        treatments = self._get_treatments()
        save_treatments(treatments, self.name)
        self.discrimination_join()

    def _get_treatments(self):
        """Generate the treatment results.
        """
        raise NotImplementedError

    def discrimination_join(self, **join_kwargs):
        """Join the results to demographic covariates and save.
        """
        discrimination_treatments_join(self.name, **join_kwargs)

    def plot(self, **kwargs):
        """Print and plot the results, comparing treatments.
        """
        treatments = load_treatments(self.name)
        compare_treatments(treatments, experiment_name=self.name, **kwargs)

    @staticmethod
    def get_experiment(
        name: str,
        *args, **kwargs
    ):
        """Factory class for generating experiment object from name.

        Args:
            name (str): Experiment name to fetch. Options include "baseline",
                "hold_harmless", "post_processing", "thresholds", "epsilon",
                "moving_average", "budget", "sampling".

        Returns:
            Experiment: The experiment object.
        """
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
    """Baseline condition only.
    """
    def _get_treatments(self):
        return {
            'baseline': self.baseline
        }


class HoldHarmless(Experiment):
    """Hold harmless and state minimum provisions.
    """
    def _get_treatments(self):
        hold_harmless = titlei_grid(
            self.saipe, Laplace,
            eps=self.eps, delta=self.delta,
            trials=self.trials, print_results=False,
            allocator_kwargs={'hold_harmless': True}
        )
        state_minimum = titlei_grid(
            self.saipe, Laplace,
            eps=self.eps, delta=self.delta,
            trials=self.trials, print_results=False,
            allocator_kwargs={'state_minimum': True}
        )
        both = titlei_grid(
            self.saipe, Laplace,
            eps=self.eps, delta=self.delta,
            trials=self.trials, print_results=False,
            allocator_kwargs={
                'hold_harmless': True,
                'state_minimum': True
            }
        )
        # ground truths are the same
        match_true(self.baseline, [hold_harmless, state_minimum, both])

        # save treatments to file for later
        return {
            'No provisions (baseline)': self.baseline,
            'Hold harmless only': hold_harmless,
            'State minimum only': state_minimum,
            'Both provisions': both
        }


class PostProcessing(Experiment):
    """Post-processing variations: none, clipping, and clipping + rounding.
    """
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
    """Moving average inputs, with multiple lag conditions.
    """
    def __init__(
        self,
        name: str,
        truth: str = "average",
        **kwargs
    ):
        """
        Args:
            truth (str, optional): Baseline strategy to use. "average" uses
                the 5-year average as the baseline. Defaults to "average".

        Raises:
            Exception: _description_
        """
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
    """Thresholding experiments.
    """
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
    """Budget experiments.
    """
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
    """Privacy parameter experiments.
    """
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
    """Sampling error experiments.
    """
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
