import pandas as pd
from typing import Tuple
import numpy as np
from diffprivlib.mechanisms import Laplace as LaplaceMech
from diffprivlib.mechanisms import GaussianAnalytic as GaussianMech
from diffprivlib.accountant import BudgetAccountant


class Mechanism:
    """
    a class for the different privacy mechanisms we might employ to compute
    poverty estimates
    """
    def __init__(
        self, sensitivity=2.0, round=False, clip=True
    ):
        self.sensitivity = sensitivity
        self.round = round
        self.clip = clip

    def poverty_estimates(
        self, pop_total, children_total, children_poverty
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns dataframe for children in poverty, children total, and total
        population indexed by district ID.
        """
        raise NotImplementedError

    def post_processing(self, count):
        if self.round:
            count = np.round(count)
        if self.clip:
            count = np.clip(count, 0, None)
        return count


class GroundTruth(Mechanism):
    def __init__(self, *args, **kwargs):
        pass

    def poverty_estimates(
        self, pop_total, children_total, children_poverty
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return pop_total, children_total, children_poverty


class DummyMechanism():
    def randomise(self, x):
        return x


class DiffPriv(Mechanism):
    def __init__(
        self, epsilon, delta, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = None
        # for advanced composition
        self.accountant = BudgetAccountant(delta=self.delta)
        self.accountant.set_default()

    def poverty_estimates(
        self, pop_total, children_total, children_poverty
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if self.mechanism is None:
            raise NotImplementedError

        pop_total = pop_total.apply(self.mechanism.randomise)
        children_total = children_total.apply(self.mechanism.randomise)
        children_poverty = children_poverty.apply(self.mechanism.randomise)

        # print("After estimation, privacy acc:", self.accountant.total())
        # no negative values, please
        # also rounding counts - post-processing
        return self.post_processing(pop_total),\
            self.post_processing(children_total),\
            self.post_processing(children_poverty)


class Laplace(DiffPriv):
    """
    Following Abowd & Schmutte (2019), return $\\hat{E}_l = E_l + e_l$,
    where $e_l \\sim Laplace(1/\\epsilon)$.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanism = LaplaceMech(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.sensitivity
        )


class Gaussian(DiffPriv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanism = GaussianMech(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.sensitivity
        )


class Sampled(Mechanism):
    def __init__(
        self,
        *args,
        multiplier=1.0, distribution="gaussian",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # these are fixed, because sampling error
        # is theoretically immutable by algo means.
        # reported estimates are non-negative integers.
        self.clip = True
        self.round = True
        self.multiplier = multiplier
        self.distribution = distribution

    def poverty_estimates(
        self, pop_total, children_total, children_poverty, cv
    ):
        if self.distribution == "gaussian":
            noised = np.random.normal(
                children_poverty,  # mean
                children_poverty * cv * self.multiplier  # variance
            )
        elif self.distribution == "laplace":
            noised = np.random.laplace(
                # mean
                children_poverty,
                # variance is 2b^2, so b = np.sqrt ( 1/2 variance )
                np.sqrt(0.5 * children_poverty * cv * self.multiplier)
            )
        else:
            raise ValueError(
                f"{self.distribution} is not a valid distribution."
            )
        children_poverty = np.clip(
            noised,
            0,
            None
        )
        return self.post_processing(pop_total), \
            self.post_processing(children_total), \
            self.post_processing(children_poverty)
