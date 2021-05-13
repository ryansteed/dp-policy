import pandas as pd
from typing import Tuple
import numpy as np
from diffprivlib.mechanisms import Laplace as LaplaceMech
from diffprivlib.mechanisms import GaussianAnalytic as GaussianMech
from diffprivlib.accountant import BudgetAccountant


class Mechanism:
    """
    a class for the different privacy mechanisms we might employ to compute poverty estimates
    """
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta
        # for advanced composition
        self.accountant = BudgetAccountant(delta=self.delta)
        self.accountant.set_default()

    def poverty_estimates(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns dataframe for children in poverty, children total, and total population indexed by district ID.
        """
        raise NotImplementedError


class GroundTruth(Mechanism):
    """
    example mech that just returns the ground truth SAIPE estimates
    """
    def __init__(self, saipe):
        self.saipe = saipe

    def poverty_estimates(self):
        return (self.saipe[key] for key in ["Estimated Total Population", "Estimated Population 5-17", "Estimated number of relevant children 5 to 17 years old in poverty who are related to the householder"])


class DiffPriv(Mechanism):
    def __init__(self, saipe, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saipe = saipe
        self.mechanism = None

    def poverty_estimates(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.mechanism is None:
            raise NotImplementedError

        pop_total = self.saipe["Estimated Total Population"].apply(self.mechanism.randomise)
        children_total = self.saipe["Estimated Population 5-17"].apply(self.mechanism.randomise)
        children_poverty = self.saipe["Estimated number of relevant children 5 to 17 years old in poverty who are related to the householder"].apply(self.mechanism.randomise)

        # print("After estimation, privacy acc:", self.accountant.total())
        # no negative values, please
        return np.clip(pop_total, 0, None), np.clip(children_total, 0, None), np.clip(children_poverty, 0, None)


class Laplace(DiffPriv):
    """
    Following Abowd & Schmutte (2019), return \hat{E}_l = E_l + e_l, where e_l \sim Laplace(1/\epsilon). 

    Recall that the sensitivity of the counts here is simply 1.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanism = LaplaceMech(epsilon=self.epsilon, delta=self.delta, sensitivity=1.0)


class Gaussian(DiffPriv):
    ""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanism = GaussianMech(epsilon=self.epsilon, delta=self.delta, sensitivity=1.0)
