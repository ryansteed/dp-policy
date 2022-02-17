import pandas as pd
from dp_policy.titlei.evaluation import \
    discrimination_treatments_join, save_treatments, load_treatments, \
    match_true, compare_treatments
from dp_policy.api import titlei_grid as test_params
from dp_policy.titlei.mechanisms import Laplace
from dp_policy.titlei.utils import get_inputs
from dp_policy.titlei.thresholders import \
    AverageThresholder, RepeatThresholder, DummyThresholder, MOEThresholder


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
        if baseline == "cached":
            print("Using cached baseline...")
            self.baseline = load_treatments("baseline")['baseline']
        elif baseline is None:
            print("Generating baseline...")
            self.baseline = test_params(
                self.saipe, Laplace,
                eps=eps, delta=delta,
                trials=self.trials,
                print_results=False,
                allocator_kwargs={'verbose': False}
            )
            print("Using given baseline...")
            save_treatments({'baseline': self.baseline}, "baseline")
        else:
            self.baseline = baseline
        self.saipe = get_inputs(year)

    def run(self):
        treatments = self._get_treatments()
        save_treatments(treatments, self.name)
        self.discrimination_join()

    def _get_treatments(self):
        raise NotImplementedError

    def discrimination_join(self, **join_kwargs):
        discrimination_treatments_join(self.name, **join_kwargs)

    def plot(self):
        treatments = load_treatments(self.name)
        compare_treatments(treatments, experiment_name=self.name)

    @staticmethod
    def get_experiment(name, *args, **kwargs):
        experiments = {
            'hold_harmless': HoldHarmless,
            'post_processing': PostProcessing,
            'thresholds': Thresholds,
            'epsilon': Epsilon,
            'budget': Budget
        }
        Exp = experiments.get(name)
        if Exp is None:
            raise ValueError(f"{name} not a supported experiment name.")
        return Exp(name, *args, **kwargs)


class HoldHarmless(Experiment):
    def _get_treatments(self):
        hold_harmless = test_params(
            self.saipe, Laplace,
            eps=self.eps, delta=self.delta,
            trials=self.trials, print_results=False,
            allocator_kwargs={'hold_harmless': True}
        )
        # ground truths are the same
        match_true(self.baseline, [hold_harmless])

        # save treatments to file for later
        return {
            'No hold harmless': self.baseline,
            'Hold harmless': hold_harmless
        }


class PostProcessing(Experiment):
    def _get_treatments(self):
        none = test_params(
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
        rounding = test_params(
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
            'Clipping': clipping,
            'Rounding': rounding
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
            test_params(
                get_inputs(2021, avg_lag=i+1),
                Laplace,
                eps=self.eps,
                delta=self.delta,
                trials=self.trials,
                print_results=False
            )
            for i in range(4)
        ]
        # pickle.dump(averaged, open("../results/averaged.pkl", 'wb'))
        match_true(averaged[-1], [single_year] + averaged[:-1])
        return {
            'lag_0': single_year,
            **{f"lag_{i+1}": a for i, a in enumerate(averaged)}
        }


class Thresholds(Experiment):
    def _get_treatments(self):
        # hard threshold
        hard = self.baseline

        # use average
        averaged = test_params(
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
        repeat2 = test_params(
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
        repeat3 = test_params(
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
        none = test_params(
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
        moe_01 = test_params(
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
            'none': none,
            'hard': hard,
            'average': averaged,
            'repeat_years=2': repeat2,
            'repeat_years=3': repeat3,
            'moe_alpha=0.1': moe_01
        }


class Epsilon(Experiment):
    def _get_treatments(self):
        return {
            f"eps={e}": test_params(
                self.saipe, Laplace,
                eps=[e], delta=self.delta,
                trials=self.trials, print_results=False
            ) if e not in self.eps
            else self.baseline.loc[pd.IndexSlice[:, 0.0, e, :, :], :].copy()
            for e in [1e-3, 1e-2, 1e-1, 1, 10]
        }

    def discrimination_join(self):
        return super().discrimination_join(epsilon=None)


class Budget(Experiment):
    def _get_treatments(self):
        if len(self.eps > 0):
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

        usual_budget = test["official_total_alloc"]\
            .groupby(["State FIPS Code", "District ID"]).first().sum()
        print("baseline budget: {:.1e}".format(usual_budget))
        print(budgets)

        treatments = {
            name: test_params(
                self.saipe, Laplace,
                eps=[e], delta=self.delta,
                trials=self.trials, print_results=False,
                allocator_kwargs={
                    'appropriation': round(budget + usual_budget, 2)
                }
            ) for name, budget in budgets.items()
        }
        match_true(self.baseline, list(treatments.values()))
        treatments["Original appropriation"] = test

        return treatments
