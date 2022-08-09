import numpy as np
import scipy.stats as stats

from dp_policy.titlei.utils import data
from dp_policy.titlei.utils import get_inputs

import pandas as pd


class Threshold:
    """Class for eligibility thresholds.
    """
    def __init__(
        self,
        t: float,
        prop: bool = False
    ):
        """
        Args:
            t (float): The threshold.
            prop (bool, optional): Whether this is a proportional threshold.
                Defaults to False.
        """
        self.t = t
        self.prop_threshold = prop

    def get_mask(
        self,
        eligible: pd.Series,
        total: pd.Series
    ) -> pd.Series:
        """Decide which LEAs are eligible.

        Args:
            eligible (pd.Series): Number of eligible children, keyed by LEA.
            total (pd.Series): Number of total children, keyed by LEA.

        Returns:
            pd.Series: Booleans indicating which LEAs are eligible.
        """
        return Threshold._mask(
            self.t,
            eligible, total,
            self.prop_threshold
        )

    @staticmethod
    def _mask(t, eligible, total, prop):
        if prop:
            return eligible / total > t
        else:
            return eligible > t


class MOEThreshold(Threshold):
    """Thresholder that decreases threshold by the margin of error.
    """
    def __init__(self, cv: pd.Series, alpha: float, *args, **kwargs) -> None:
        """
        Args:
            cv (pd.Series): Coefficients of variation, keyed by LEA.
            alpha (float): Confidence level for MoE.
        """
        super().__init__(*args, **kwargs)
        self.cv = cv
        self.alpha = alpha

    def get_mask(self, eligible, total):
        t = self.t * (1 - self.cv * stats.norm.ppf(1 - self.alpha / 2))
        return Threshold._mask(
            t,
            eligible, total,
            self.prop_threshold
        )

    @staticmethod
    def _from_base(base, *init_args):
        return MOEThreshold(
            *init_args,
            base.t,
            prop=base.prop_threshold,
        )


class Thresholder:
    """Class for implementing thresholds.
    """
    def process(
        self,
        y,
        in_poverty, total, thresholds,
        comb_func=np.logical_and.reduce
    ):
        raise NotImplementedError

    def set_cv(self, cv):
        self.cv = cv


class DummyThresholder(Thresholder):
    """Doesn't apply the threshold.
    """
    def process(
        self,
        y,
        *args,
        **kwargs
    ):
        return y, np.ones(y.shape)


class HardThresholder(Thresholder):
    """Hard threshold (default).
    """
    def process(
        self,
        y,
        in_poverty, total, thresholds,
        comb_func=np.logical_and.reduce
    ):
        masks = self._masks(in_poverty, total, thresholds)
        return self._enforce(comb_func(masks), y.copy())

    def _enforce(self, eligible, y):
        y.loc[~eligible, :] = 0.0
        return y, eligible

    def _masks(self, in_poverty, total, thresholds):
        return [t.get_mask(in_poverty, total) for t in thresholds]


class MOEThresholder(HardThresholder):
    """MoE thresholder.
    """
    def __init__(self, alpha: float = 0.1) -> None:
        """
        Args:
            alpha (float, optional): Confidence level. Defaults to 0.1.
        """
        super().__init__()
        self.alpha = 0.1

    def process(
        self,
        y, in_poverty, total, thresholds,
        **kwargs
    ):
        if self.cv is None:
            raise ValueError(
                "Missing coefficients of varation - call set_cv first."
            )
        thresholds_moe = [
            MOEThreshold._from_base(
                threshold,
                self.cv,
                self.alpha
            )
            for threshold in thresholds
        ]
        return super().process(
            y, in_poverty, total,
            thresholds_moe,
            **kwargs
        )


class PastThresholder(HardThresholder):
    """Thresholds based on past data.
    """
    def __init__(
        self,
        year: int,
        lag: int
    ) -> None:
        """
        Args:
            year (int): Current year data.
            lag (int): How many years back to include.
        """
        self.prior_estimates = None
        self.year = year
        self.lag = lag

    @staticmethod
    def _get_count_past(past_df, count):
        return past_df[count.name].reindex(count.index)

    def set_prior_estimates(self, *data_args, **data_kwargs):
        raise NotImplementedError


class RepeatThresholder(PastThresholder):
    """Thresholds based on repeated eligibility.
    """
    def set_prior_estimates(self, *data_args, **data_kwargs):
        self.prior_estimates = [
            data(
                get_inputs(self.year-i, verbose=False),
                *data_args,
                **data_kwargs
            )
            # first entry is this year, ignore
            for i in range(1, self.lag)
        ]

    def process(
        self,
        y, in_poverty, total, thresholds,
        comb_func=np.logical_and.reduce
    ):
        y = y.copy()
        masks = self._masks(in_poverty, total, thresholds)
        # only count as ineligible if ineligible for {lag}
        # years in a row
        eligible_prior = np.logical_or.reduce([
            comb_func(self._masks(
                PastThresholder._get_count_past(past_df, in_poverty),
                PastThresholder._get_count_past(past_df, total),
                thresholds
            ))
            for past_df in self.prior_estimates
        ])
        return self._enforce(
            comb_func(masks) | eligible_prior,
            y
        )


class AverageThresholder(PastThresholder):
    """Thresholds based on a moving average.
    """
    def set_prior_estimates(self, *data_args, **data_kwargs):
        self.prior_estimates = data(
            get_inputs(self.year, avg_lag=self.lag, verbose=False),
            *data_args,
            **data_kwargs
        )

    def process(
        self,
        y, in_poverty, total, thresholds,
        **kwargs
    ):
        assert self.prior_estimates is not None, "Test"
        in_poverty_avg = PastThresholder._get_count_past(
            self.prior_estimates, in_poverty
        )
        total_avg = PastThresholder._get_count_past(
            self.prior_estimates, total
        )
        return super().process(
            y, in_poverty_avg, total_avg,
            thresholds,
            **kwargs
        )
