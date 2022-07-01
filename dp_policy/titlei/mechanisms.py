import pandas as pd
from typing import Tuple
import numpy as np
from diffprivlib.mechanisms import Laplace as LaplaceMech
from diffprivlib.mechanisms import GaussianAnalytic as GaussianMech
from diffprivlib.accountant import BudgetAccountant


class Mechanism:
    """
    A class for the different privacy mechanisms we employ to compute
    poverty estimates.
    """
    def __init__(
        self, sensitivity=2.0, round=False, clip=True, noise_total=False
    ):
        self.sensitivity = sensitivity
        self.round = round
        self.clip = clip
        self.noise_total = noise_total

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
    """No randomization.
    """
    def __init__(self, *args, **kwargs):
        pass

    def poverty_estimates(
        self, pop_total, children_total, children_poverty
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return pop_total, children_total, children_poverty


class DummyMechanism():
    """No randomization.
    """
    def randomise(self, x):
        return x


class DiffPriv(Mechanism):
    """Differentially private mechanisms wrapping `diffprivlib`.
    """
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

        # NOTE: as of 3/21, by default only adding noise to poverty estimate
        # (for consistency with sampling, where est. var. is unavailable)
        children_poverty = children_poverty.apply(self.mechanism.randomise)
        if self.noise_total:
            children_total = children_total.apply(self.mechanism.randomise)

        # print("After estimation, privacy acc:", self.accountant.total())
        # no negative values, please
        # also rounding counts - post-processing
        return self.post_processing(pop_total),\
            self.post_processing(children_total),\
            self.post_processing(children_poverty)


class Laplace(DiffPriv):
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
    """Mechanism for simulating sampling errors.
    """
    def __init__(
        self,
        *args,
        multiplier: float = 1.0,
        distribution: str = "gaussian",
        **kwargs
    ):
        """
        Args:
            multiplier (float, optional): Scales sampling noise by a constant.
                Defaults to 1.0.
            distribution (str, optional): Distribution of sampling noise.
                Supported options are 'gaussian' and 'laplace'. Defaults to
                "gaussian".
        """
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
        children_poverty = self._noise(children_poverty, cv)
        if self.noise_total:
            # NOTE: assuming CVs are same for total children.
            # This is beyond Census guidance.
            children_total = self._noise(children_total, cv)

        return self.post_processing(pop_total), \
            self.post_processing(children_total), \
            self.post_processing(children_poverty)

    def _noise(self, count, cv):
        if self.distribution == "gaussian":
            noised = np.random.normal(
                count,  # mean
                count * cv * self.multiplier  # stderr
            )
        elif self.distribution == "laplace":
            noised = np.random.laplace(
                # mean
                count,
                # variance is 2b^2 = (count * cv)^2
                # b = count * cv * sqrt(1/2)
                np.sqrt(0.5) * count * cv * self.multiplier
            )
        else:
            raise ValueError(
                f"{self.distribution} is not a valid distribution."
            )
        return np.clip(
            noised,
            0,
            None
        )
