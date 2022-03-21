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
        appropriation=None,
        verbose=False
    ):
        self.estimates = estimates
        self.prefixes = prefixes
        self.congress_cap = congress_cap
        self.adj_sppe_bounds = adj_sppe_bounds
        self.adj_sppe_bounds_efig = adj_sppe_bounds_efig
        self.appropriation_total = appropriation
        self.verbose = verbose

    def allocations(
        self,
        normalize=True
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


class Authorizer(Allocator):
    def grant_types(self):
        raise NotImplementedError

    def calc_total(self):
        for prefix in self.prefixes:
            self.estimates[f"{prefix}_grant_total"] = self.estimates[[
                f"{prefix}_grant_{grant_type}"
                for grant_type in self.grant_types()
            ]].sum(axis=1)
        return self.estimates

    def allocations(
        self, normalize=True, **kwargs
    ) -> pd.DataFrame:
        super().allocations(**kwargs)
        if normalize:
            for grant_type in self.grant_types():
                for prefix in self.prefixes:
                    appropriation = \
                        self._calc_appropriation_total(grant_type)
                    self._normalize(grant_type, prefix, appropriation)
            self.calc_total()
            if self.verbose:
                total_approp = self.estimates[[
                    f"official_{grant_type}_alloc"
                    for grant_type in self.grant_types()
                ]].sum().sum()
                print(
                    "After normalization, appropriation is",
                    total_approp,
                    "and true allocation is",
                    self.estimates.true_grant_total.sum()
                )
        return self.estimates

    def _calc_appropriation_total(self, grant_type):
        appropriation = self.estimates[f"official_{grant_type}_alloc"].sum()
        if self.appropriation_total is not None:
            if self.verbose:
                print("Usual appropriation:", appropriation)
                print(
                    "Usual total:",
                    self.estimates[f"official_total_alloc"].sum()
                )
                print(
                    self.appropriation_total /
                    self.estimates[f"official_total_alloc"].sum()
                )
                print("New appropriation:", (
                    appropriation * self.appropriation_total /
                    self.estimates[f"official_total_alloc"].sum()
                ))
            # scale appropriation to total budget
            return (
                appropriation * self.appropriation_total /
                self.estimates[f"official_total_alloc"].sum()
            )
        return appropriation

    def _normalize(
        self, grant_type, prefix, appropriation, hold_harmless=None
    ):
        # -- DEPRECATED by new official file --
        # get this year's budget
        # true_allocs = get_allocation_data("../data/titlei-allocations_20")
        # actual_budget = true_allocs["HistAlloc"].sum()

        # print(actual_budget)
        # print(len(true_allocs))

        if hold_harmless is None:
            hold_harmless = np.zeros(len(self.estimates)).astype(bool)

        # available budget is the full budget minus hold harmless districts
        remaining_budget = appropriation - self.estimates.loc[
            hold_harmless, f"{prefix}_grant_{grant_type}"
        ].sum()
        current_budget = \
            self.estimates[f"{prefix}_grant_{grant_type}"].sum()
        # redistribute the remaining budget between non-harmless districts
        self.estimates.loc[
            ~hold_harmless, f"{prefix}_grant_{grant_type}"
        ] = \
            Authorizer.normalize_to_budget(
                self.estimates.loc[
                    ~hold_harmless, f"{prefix}_grant_{grant_type}"
                ], remaining_budget
            )
        if self.verbose:
            print(
                f"{current_budget} authorized for {grant_type} reduced "
                f"to {appropriation} allocated."
            )

    @staticmethod
    def normalize_to_budget(authorizations, total_budget):
        """Scale authorizations proportional to total federal budget.

        Args:
            total_budget (int): Estimated total budget for Title I this year.
        """
        return authorizations / authorizations.sum() * total_budget


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
            "targeted"
        )

    def calc_auth(self):
        # calc adj. SPPE
        adj_sppe, _ = self.adj_sppe()
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

        self.calc_total()

    def _hold_harmless(self):
        if self.verbose:
            print("Applying hold harmless")

        # load last year's allocs - watch out for endogeneity
        # get this year's budget
        for grant_type in self.grant_types():
            alloc_previous = \
                self.estimates[f"official_{grant_type}_hold_harmless"]
            appropriation = self._calc_appropriation_total(grant_type)
            for prefix in self.prefixes:
                harmless_rate = SonnenbergAuthorizer._hold_harmless_rate(
                    self.estimates[f"{prefix}_children_eligible"] /
                    self.estimates[f"{prefix}_children_total"]
                )
                self._hold_harmless_recursive(
                    0,
                    np.zeros(len(self.estimates)).astype(bool),
                    prefix,
                    grant_type,
                    appropriation,
                    harmless_rate,
                    alloc_previous
                )

        self.calc_total()
        if self.verbose:
            total_approp = self.estimates[[
                f"official_{grant_type}_alloc"
                for grant_type in self.grant_types()
            ]].sum().sum()
            print(
                "After hold harmless, appropriation is",
                total_approp,
                "and true allocation is",
                self.estimates.true_grant_total.sum()
            )

    def _hold_harmless_recursive(
        self,
        depth,
        hist_harmless,
        prefix,
        grant_type,
        appropriation,
        harmless_rate,
        alloc_previous,
        max_depth=10
    ):
        hold_harmless = self._excessive_loss(
            prefix, grant_type, alloc_previous, harmless_rate
        )
        if not hold_harmless.any():
            return
        if self.verbose and depth > 0:
            print(
                "Hold harmless iter", depth,
                f"- {hold_harmless.sum()} hold harmless districts remaining"
            )
        if depth > max_depth:
            print(
                f"[WARN]: {hold_harmless.sum()} not held harmless."
                f"Could not converge after {max_depth} iterations."
            )
            return
        # limit losses to harmless rate
        hist_harmless = hold_harmless | hist_harmless
        self.estimates.loc[hold_harmless, f"{prefix}_grant_{grant_type}"] = \
            alloc_previous * harmless_rate
        self._normalize(
            grant_type, prefix, appropriation,
            hold_harmless=hist_harmless
        )
        return self._hold_harmless_recursive(
            depth+1,
            hist_harmless,
            prefix,
            grant_type,
            appropriation,
            harmless_rate,
            alloc_previous,
            max_depth=max_depth
        )

    def _excessive_loss(
        self, prefix, grant_type, alloc_previous, harmless_rate
    ):
        return (
            self.estimates[f"{prefix}_grant_{grant_type}"] <
            harmless_rate * alloc_previous
        )

    @staticmethod
    def _hold_harmless_rate(prop_eligible):
        return np.where(
            prop_eligible < 0.15,
            0.85,
            np.where(
                prop_eligible < 0.3,
                0.9,
                0.95
            )
        )
