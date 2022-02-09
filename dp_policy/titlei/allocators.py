from dp_policy.titlei.utils import \
    weighting, get_allocation_data
from dp_policy.titlei.thresholders import Threshold, HardThresholder

import numpy as np
import pandas as pd
import scipy.stats as stats


class Allocator:
    def __init__(
        self,
        estimates,
        prefixes=("true", "est", "dp", "dpest"),
        congress_cap=0.4,
        adj_sppe_bounds=[0.32, 0.48],
        adj_sppe_bounds_efig=[0.34, 0.46],
        verbose=False
    ):
        self.estimates = estimates
        self.prefixes = prefixes
        self.congress_cap = congress_cap
        self.adj_sppe_bounds = adj_sppe_bounds
        self.adj_sppe_bounds_efig = adj_sppe_bounds_efig
        self.verbose = verbose

    def allocations(
        self
    ) -> pd.DataFrame:
        self.calc_auth()
        return self.estimates

    def calc_auth(self):
        """
        Appends the allocated grants as columns to the estimates DataFrame.

        Must generate at least `true_grant_total` and `est_grant_total`.

        returns:
            pd.DataFrame: current estimates
        """
        raise NotImplementedError

    def normalize(self):
        """Normalize authorization amounts to allocation amounts.
        """

    def adj_sppe(self):
        """Calculate adjusted SPPE using Sonnenberg, 2016 pg. 18 algorithm.
        """
        # Get baseline average across all 50 states and territories
        average = np.round(
            self.estimates.sppe.groupby("State FIPS Code").first().mean(),
            decimals=2
        )
        # Each state’s and each territory’s SPPE is multiplied by the
        # congressional cap and rounded to the second decimal place
        # (for dollars and cents).
        scaled = np.round(self.estimates.sppe * self.congress_cap, decimals=2)
        # No state recieves above/below the bounds set by law
        adj_sppe_trunc = scaled.clip(
            # bound by some % of the average, given in the law - round to cents
            *np.round(np.array(self.adj_sppe_bounds)*average, decimals=2)
        )
        adj_sppe_efig = scaled.clip(
            # bound %s are different for EFIG
            *np.round(np.array(self.adj_sppe_bounds_efig)*average, decimals=2)
        )
        return adj_sppe_trunc, adj_sppe_efig


class Authorizer(Allocator):
    def grant_types(self):
        raise NotImplementedError

    def allocations(
        self, normalize=True, **kwargs
    ) -> pd.DataFrame:
        super().allocations(**kwargs)
        if normalize:
            for prefix in self.prefixes:
                self._normalize("total", prefix)
        return self.estimates

    def _normalize(self, grant_type, prefix, hold_harmless=None):
        # -- DEPRECATED by new official file --
        # get this year's budget
        # true_allocs = get_allocation_data("../data/titlei-allocations_20")
        # actual_budget = true_allocs["HistAlloc"].sum()

        # print(actual_budget)
        # print(len(true_allocs))

        appropriation = self.estimates[f"official_{grant_type}_alloc"].sum()

        if hold_harmless is None:
            hold_harmless = np.zeros(len(self.estimates)).astype(bool)

        # available budget is the full budget minus hold harmless districts
        remaining_budget = appropriation - self.estimates.loc[
            hold_harmless, f"{prefix}_grant_{grant_type}"
        ].sum()
        current_budget = \
            self.estimates[f"{prefix}_grant_{grant_type}"].sum()
        # only rescale total amount - other amounts remain the same
        # (rescaling happens after amounts are summed)
        authorization_amounts = [
            f"{prefix}_grant_{grant_type}"
        ]

        # redistribute the remaining budget between non-harmless districts
        self.estimates.loc[~hold_harmless, authorization_amounts] = \
            self.estimates.loc[
                ~hold_harmless, authorization_amounts
            ].apply(
                lambda x: Authorizer.normalize_to_budget(
                    x, remaining_budget
                )
            )
        if self.verbose:
            print(
                f"{current_budget} authorized reduced "
                f"to {appropriation} allocated."
            )

    @staticmethod
    def normalize_to_budget(authorizations, total_budget):
        """Scale authorizations proportional to total federal budget.

        Args:
            total_budget (int): Estimated total budget for Title I this year.
        """
        return authorizations / authorizations.sum() * total_budget


class AbowdAllocator(Allocator):
    """
    As described in https://arxiv.org/pdf/1808.06303.pdf
    """
    def grant_types(self):
        return (
            "total"
        )

    def calc_auth(self):
        adj_sppe, _ = self.adj_sppe()

        self.estimates["adj_sppe"] = adj_sppe
        for prefix in self.prefixes:
            self.estimates[f"{prefix}_grant_total"] = \
                adj_sppe * self.estimates[f"{prefix}_children_eligible"]

        return self.estimates


class SonnenbergAuthorizer(Authorizer):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.hold_harmless = kwargs.pop('hold_harmless', False)
        self.thresholder = kwargs.pop('thresholder', HardThresholder())
        super().__init__(*args, **kwargs)

    def allocations(
        self, **kwargs
    ) -> pd.DataFrame:
        super().allocations(**kwargs)
        if self.hold_harmless:
            self._hold_harmless()
        return self.estimates

    def grant_types(self):
        return (
            "basic",
            "concentration",
            "targeted",
            "total"
        )

    def calc_auth(self):
        # calc adj. SPPE
        adj_sppe, adj_sppe_efig = self.adj_sppe()
        self.thresholder.set_cv(self.estimates.cv)

        # calculate grant amounts for true/randomized values
        for prefix in self.prefixes:
            # BASIC GRANTS
            # authorization calculation
            self.estimates[f"{prefix}_grant_basic"] = \
                self.estimates[f"{prefix}_children_eligible"] * adj_sppe
            # For basic grants, LEA must have
            # >10 eligible children
            # AND >2% eligible

            # count_eligible = \
            #     self.estimates[f"{prefix}_children_eligible"] > 10
            # share_eligible = (
            #     self.estimates[f"{prefix}_children_eligible"]
            #     / self.estimates[f"{prefix}_children_total"]
            # ) > 0.02
            # self.estimates[f"{prefix}_eligible_basic"] = \
            #     count_eligible & share_eligible
            # self.estimates.loc[
            #     ~self.estimates[f"{prefix}_eligible_basic"],
            #     f"{prefix}_grant_basic"
            # ] = 0.0

            grants, eligible = \
                self.thresholder.process(
                    self.estimates[f"{prefix}_grant_basic"],
                    self.estimates[f"{prefix}_children_eligible"],
                    self.estimates[f"{prefix}_children_total"],
                    [
                        Threshold(10, prop=False),
                        Threshold(0.02, prop=True)
                    ],
                    comb_func=np.logical_and.reduce
                )
            self.estimates.loc[:, f"{prefix}_grant_basic"] = grants
            self.estimates.loc[:, f"{prefix}_eligible_basic"] = eligible

            # CONCENTRATION GRANTS
            # For concentration grants, LEAs must meet basic eligibility
            # AND have either
            # a) >6500 eligible
            # b) 15% of pop. is eligible
            self.estimates[f"{prefix}_grant_concentration"] = \
                self.estimates[f"{prefix}_grant_basic"]

            # count_eligible = \
            #     self.estimates[f"{prefix}_children_eligible"] > 6500
            # prop = self.estimates[f"{prefix}_children_eligible"] \
            #     / self.estimates[f"{prefix}_children_total"]
            # prop_eligible = prop > 0.15
            # self.estimates[f"{prefix}_eligible_concentration"] = \
            #     self.estimates[f"{prefix}_eligible_basic"] & \
            #     (count_eligible | prop_eligible)
            # self.estimates.loc[
            #     ~self.estimates[f"{prefix}_eligible_concentration"],
            #     f"{prefix}_grant_concentration"
            # ] = 0.0

            grants, eligible = \
                self.thresholder.process(
                    self.estimates[f"{prefix}_grant_concentration"],
                    self.estimates[f"{prefix}_children_eligible"],
                    self.estimates[f"{prefix}_children_total"],
                    [
                        Threshold(6500, prop=False),
                        Threshold(0.15, prop=True)
                    ],
                    comb_func=lambda x:
                        self.estimates[f"{prefix}_eligible_basic"]
                        & np.logical_or.reduce(x)
                )
            self.estimates.loc[:, f"{prefix}_grant_concentration"] = grants
            self.estimates.loc[:, f"{prefix}_eligible_concentration"] = \
                eligible

            # TARGETED GRANTS
            # weighted by an exogenous step function - see documentation
            weighted_eligible = self.estimates[[
                f"{prefix}_children_eligible", f"{prefix}_children_total"
            ]].apply(
                lambda x: weighting(x[0], x[1]),
                axis=1
            )
            self.estimates[
                f"{prefix}_grant_targeted"
            ] = weighted_eligible * adj_sppe
            # for targeted grants, LEAs must:
            # meet basic eligibility AND have >5% eligible

            # count_eligible = \
            #     self.estimates[f"{prefix}_children_eligible"] > 10
            # share_eligible = self.estimates[f"{prefix}_children_eligible"] \
            #     / self.estimates[f"{prefix}_children_total"]
            # prop_eligible = share_eligible > 0.05
            # self.estimates[f"{prefix}_eligible_targeted"] = \
            #     count_eligible & prop_eligible
            # self.estimates.loc[
            #     ~self.estimates[f"{prefix}_eligible_targeted"],
            #     f"{prefix}_grant_targeted"
            # ] = 0.0

            grants, eligible = \
                self.thresholder.process(
                    self.estimates[f"{prefix}_grant_targeted"],
                    self.estimates[f"{prefix}_children_eligible"],
                    self.estimates[f"{prefix}_children_total"],
                    [
                        Threshold(10, prop=False),
                        Threshold(0.05, prop=True)
                    ],
                    comb_func=np.logical_and.reduce
                )
            self.estimates.loc[:, f"{prefix}_grant_targeted"] = grants
            self.estimates.loc[:, f"{prefix}_eligible_targeted"] = eligible

            # EFIG
            # TODO

            # clip lower bound to zero
            for grant_type in ["basic", "concentration", "targeted"]:
                self.estimates.loc[
                    self.estimates[f"{prefix}_grant_{grant_type}"] < 0.0,
                    f"{prefix}_grant_{grant_type}"
                ] = 0.0

        self.estimates = SonnenbergAuthorizer.calc_total(
            self.estimates, self.prefixes
        )

    @staticmethod
    def calc_total(results, prefixes):
        for prefix in prefixes:
            results[f"{prefix}_grant_total"] = \
                results[f"{prefix}_grant_basic"] + \
                results[f"{prefix}_grant_concentration"] + \
                results[f"{prefix}_grant_targeted"]
        return results

    def _hold_harmless(self):
        # load last year's allocs - watch out for endogeneity
        # get this year's budget
        true_allocs = get_allocation_data(
            "../data/titlei-allocations_19",
            header=2
        ).rename(columns={
            'HistAlloc': 'alloc_2019'
        })
        self.estimates = self.estimates.join(
            true_allocs["alloc_2019"]
        )
        for prefix in self.prefixes:
            i = 0
            hist_harmless = np.zeros(len(self.estimates)).astype(bool)
            while SonnenbergAuthorizer._excessive_loss(
                self.estimates[f"{prefix}_grant_total"],
                self.estimates["alloc_2019"]
            ).any():
                hold_harmless = SonnenbergAuthorizer._excessive_loss(
                    self.estimates[f"{prefix}_grant_total"],
                    self.estimates["alloc_2019"]
                )
                hist_harmless = hold_harmless | hist_harmless
                i += 1
                if (i > 10):
                    print(
                        f"[WARN]: {hold_harmless.sum()} SDs not held harmless."
                        "Could not converge after 100 iterations."
                    )
                # print(i)
                # print("# hold harmless:", hold_harmless.sum())
                # print(self.estimates.index[hold_harmless].values[:2])
                # limit losses to 15%
                self.estimates.loc[hold_harmless, f"{prefix}_grant_total"] = \
                    self.estimates["alloc_2019"] * 0.85
                # renormalize and try again
                self._normalize(hold_harmless=hist_harmless)

    @staticmethod
    def _excessive_loss(this_year, last_year):
        return this_year < 0.85 * last_year
