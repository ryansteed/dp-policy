from dp_policy.titlei.utils import \
    weighting
from dp_policy.titlei.thresholders import Threshold, HardThresholder

import numpy as np
import pandas as pd

from typing import Tuple, List


class Allocator:
    def __init__(
        self,
        estimates: pd.DataFrame,
        prefixes: Tuple[str] = ("true", "est", "dp", "dpest"),
        congress_cap: float = 0.4,
        adj_sppe_bounds: List[float] = [0.32, 0.48],
        adj_sppe_bounds_efig: List[float] = [0.34, 0.46],
        appropriation: float = None,
        verbose: bool = False
    ):
        """Class for allocating Title I funds.

        Args:
            estimates (pd.DataFrame): The poverty estimates, including
                randomized estimates.
            prefixes (Tuple[str], optional): Prefixes for different kinds of
                estimates. Defaults to ("true", "est", "dp", "dpest").
            congress_cap (float, optional): Congressional cap. Defaults to 0.4.
            adj_sppe_bounds (List[float], optional): Bounds on adjusted SPPE.
                Defaults to [0.32, 0.48].
            adj_sppe_bounds_efig (List[float], optional): Bounds on adjusted
                SPPE for EFIG grants. Defaults to [0.34, 0.46].
            appropriation (float, optional): Total congressional appropriation.
                Defaults to None.
            verbose (bool, optional): Defaults to False.
        """
        self.estimates = estimates
        self.prefixes = prefixes
        self.congress_cap = congress_cap
        self.adj_sppe_bounds = adj_sppe_bounds
        self.adj_sppe_bounds_efig = adj_sppe_bounds_efig
        self.appropriation_total = appropriation
        self.verbose = verbose

    def allocations(
        self,
        normalize: bool = False
    ) -> pd.DataFrame:
        """Compute allocations.

        Args:
            normalize (bool, optional): Whether to normalize the authorization
                amounts. Defaults to False.

        Returns:
            pd.DataFrame: Estimates dataframe with computed allocations added.
        """
        self.calc_auth()
        if normalize:
            self.normalize()
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
    """An allocator that uses a normalization strategy to convert
    authorization amounts to allocation amounts.
    """
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

    def _calc_appropriation_total(
        self,
        grant_type: str
    ) -> float:
        """Using official figures, calculate appropriation for this grant type.

        Args:
            grant_type (str): The grant type. One of "basic", "concentration",
                "targeted".

        Returns:
            float: The total appropration for this grant type.
        """
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
        self,
        grant_type: str,
        prefix: str,
        appropriation: float,
        hold_harmless: pd.Series = None,
        state_minimum: bool = False
    ):
        """Normalize funding amounts, honoring special provisions.

        Args:
            grant_type (str):  The grant type. One of "basic", "concentration",
                "targeted".
            prefix (str): Prefix for estimate. One of
                ("true", "est", "dp", "dpest").
            appropriation (float): The total appropriation available.
            hold_harmless (pd.DataFrame, optional): If provided, which
                districts are held harmless. Defaults to None.
            state_minimum (bool, optional): Whether to apply the state minimum.
                Defaults to False.
        """
        if hold_harmless is None:
            hold_harmless = np.zeros(len(self.estimates)).astype(bool)

        current_budget = \
            self.estimates[f"{prefix}_grant_{grant_type}"].sum()

        # do a round of hold harmless normalization
        # (to get the alloc instead of auth amounts)
        self._normalize_segment(
                np.ones(len(self.estimates)).astype(bool),
                hold_harmless,
                grant_type, prefix,
                appropriation
            )

        if state_minimum:
            glob_min = \
                SonnenbergAuthorizer._state_minimum_global(grant_type)
            formula_children = self.estimates[f"{prefix}_children_eligible"]\
                .where(
                    self.estimates[f"{prefix}_eligible_{grant_type}"],
                    0
                )
            state_eligible = formula_children\
                .groupby("State FIPS Code").transform('sum')
            # nat'l avg per-pupil payment (???)
            napp = appropriation \
                / self.estimates[f"{prefix}_children_total"].sum()
            eligib_comp = state_eligible * 1.5 * napp
            if grant_type == "concentration":
                # for concentration, this amount is at minimum 340000
                eligib_comp = np.clip(eligib_comp, 340000, None)

            # the state minimum is the smaller of
            state_minimums = np.minimum(
                # 1) the global minimum - derived from the official
                # FY 2021 state allocs
                np.ones(len(self.estimates)) * glob_min,
                # and 2) the average of
                1/2 * (
                    # a) global minimum and
                    # b) the state's elibility count * 1.5 * avg SPPE
                    glob_min + eligib_comp
                )
            )
            # identify LEAs in states under the minimum
            under_minimum = self.estimates[f"{prefix}_grant_{grant_type}"]\
                .groupby("State FIPS Code").transform('sum') < state_minimums
            if self.verbose:
                print(
                    "These states meet the state minimum:",
                    np.unique(
                        under_minimum[under_minimum].index
                        .get_level_values("State FIPS Code").values
                    )
                )

            # first, normalize all the over-minimum states
            total_minimum = state_minimums[under_minimum]\
                .groupby("State FIPS Code").first().sum()
            self._normalize_segment(
                ~under_minimum,
                hold_harmless,
                grant_type, prefix,
                appropriation - total_minimum
            )

            # then normalize the under-minimum states, state-by-state
            for _, group in self.estimates[under_minimum].groupby(
                "State FIPS Code"
            ):
                minimum = state_minimums[group.index].iloc[0]
                self._normalize_segment(
                    self.estimates.index.isin(group.index),
                    hold_harmless,
                    grant_type, prefix,
                    minimum
                )

        if self.verbose:
            print(
                f"{current_budget} authorized for {grant_type} reduced "
                f"to {appropriation} allocated."
            )

    def _normalize_segment(
        self,
        segment: pd.Series,
        hold_harmless: pd.Series,
        grant_type: str,
        prefix: str,
        segment_appropriation: float
    ):
        """Normalize the budget for a segment of the districts.

        Args:
            segment (pd.Series): A boolean mask indicating the segment to
                normalize.
            hold_harmless (pd.Series): Which districts should be held harmless.
            grant_type (str): Grant type to normalize.
            prefix (str): Which treatment to normalize (generally "true",
                "est" or "dpest").
            segment_appropriation (float): The total budget for this segment.
        """
        # available budget is the full budget minus hold harmless districts
        remaining_budget = segment_appropriation - self.estimates.loc[
                segment & hold_harmless, f"{prefix}_grant_{grant_type}"
            ].sum()
        # redistribute the remaining budget between non-harmless districts
        self.estimates.loc[
            segment & ~hold_harmless, f"{prefix}_grant_{grant_type}"
        ] = \
            Authorizer.normalize_to_budget(
                self.estimates.loc[
                    segment & ~hold_harmless, f"{prefix}_grant_{grant_type}"
                ], remaining_budget
            )

    @staticmethod
    def normalize_to_budget(
        authorizations: pd.Series,
        total_budget: int
    ) -> pd.Series:
        """Scale authorizations proportional to total federal budget.

        Args:
            authorizations (pd.Series): Authorization amounts.
            total_budget (int): Estimated total budget for Title I this year.

        Returns:
            pd.Series: Normalized authorization amounts.
        """
        return authorizations / authorizations.sum() * total_budget


class SonnenbergAuthorizer(Authorizer):
    """Authorizer allocator described by Sonnenberg (2007). Official process
    used by Dep Ed.
    """
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.hold_harmless = kwargs.pop('hold_harmless', False)
        self.state_minimum = kwargs.pop('state_minimum', False)
        self.thresholder = kwargs.pop('thresholder', HardThresholder())
        super().__init__(*args, **kwargs)
        if self.state_minimum and self.verbose:
            print(
                "[WARN] State minimum works using 2021 data. "
                "Will be wrong for earlier years."
            )

    def allocations(
        self, **kwargs
    ) -> pd.DataFrame:
        super().allocations(**kwargs)
        if self.hold_harmless or self.state_minimum:
            self._provisions()
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

    def _provisions(self):
        """
        Apply post-formula provisions (hold harmless and state minimum).
        Achieved by recursively updating allocations until all provisions are
        satisfied.
        """
        if self.verbose:
            if self.hold_harmless:
                print("Applying hold harmless")
            if self.state_minimum:
                print("Applying state minimum")

        # load last year's allocs - watch out for endogeneity
        # get this year's budget
        for grant_type in self.grant_types():
            alloc_previous = \
                self.estimates[f"official_{grant_type}_hold_harmless"]
            appropriation = self._calc_appropriation_total(grant_type)
            for prefix in self.prefixes:
                # hold harmless
                self._harmless_rate = SonnenbergAuthorizer._hold_harmless_rate(
                    self.estimates[f"{prefix}_children_eligible"] /
                    self.estimates[f"{prefix}_children_total"]
                )
                self._provisions_recursive(
                    0,
                    prefix,
                    grant_type,
                    appropriation,
                    alloc_previous
                )

        self.calc_total()
        if self.verbose:
            total_approp = self.estimates[[
                f"official_{grant_type}_alloc"
                for grant_type in self.grant_types()
            ]].sum().sum()
            print(
                "After provision(s), appropriation is",
                total_approp,
                "and true allocation is",
                self.estimates.true_grant_total.sum()
            )

    def _provisions_recursive(
        self,
        depth: int,
        prefix: str,
        grant_type: str,
        appropriation: float,
        alloc_previous: pd.Series,
        held_harmless: bool = None,
        max_depth: int = 10
    ):
        """Apply one iteration of post-formula provisions.

        Args:
            depth (int): Current iteration of recursion.
            prefix (str): Prefix of allocation to adjust.
            grant_type (str): Grant type to adjust.
            appropriation (float): Appropriation for this grant type.
            alloc_previous (pd.Series): Previous year's allocations.
            held_harmless (bool, optional): Boolena mask for districts to hold
                harmless. Defaults to None.
            max_depth (int, optional): Maximum number of iterations to run
                before stopping. Defaults to 10.
        """
        # assume no LEAs in violation of provisions
        leas_in_violation = np.zeros(len(self.estimates)).astype(bool)

        if self.hold_harmless:
            if held_harmless is None:
                held_harmless = np.zeros(len(self.estimates)).astype(bool)
            # identify LEAs suffering excessive harm
            hold_harmless = self._excessive_loss(
                prefix, grant_type, alloc_previous, self._harmless_rate
            )
            if self.verbose and depth > 0:
                print(
                    "Hold harmless iter", depth,
                    f"- {hold_harmless.sum()} hold harm districts remaining"
                )
            # limit losses to the appropriate harmless rate
            if hold_harmless.any():
                held_harmless = held_harmless | hold_harmless
                leas_in_violation = leas_in_violation | hold_harmless
                self.estimates.loc[
                    hold_harmless, f"{prefix}_grant_{grant_type}"
                ] = alloc_previous * self._harmless_rate
                self._normalize(
                    grant_type, prefix, appropriation,
                    hold_harmless=held_harmless,
                    state_minimum=self.state_minimum
                )
        else:
            self._normalize(
                grant_type, prefix, appropriation,
                state_minimum=self.state_minimum
            )

        # once no LEAs are in violation, finish
        if not leas_in_violation.any():
            return

        if depth >= max_depth:
            if self.hold_harmless:
                print(f"[WARN]: {hold_harmless.sum()} not held harmless.")
            print(
                f"[WARN] Could not converge after {max_depth} iterations."
            )
            return

        return self._provisions_recursive(
            depth+1,
            prefix,
            grant_type,
            appropriation,
            alloc_previous,
            held_harmless=held_harmless,
            max_depth=max_depth
        )

    def _excessive_loss(
        self,
        prefix: str,
        grant_type: str,
        alloc_previous: pd.Series,
        harmless_rate: pd.Series
    ) -> pd.Series:
        """Determine whether a district has suffered excessive loss.

        Returns:
            pd.Series: Boolean mask for each district indicating whether it
                has suffered excessive loss compared to the previous
                year's allocation.
        """
        return (
            self.estimates[f"{prefix}_grant_{grant_type}"] <
            harmless_rate * alloc_previous
        )

    @staticmethod
    def _hold_harmless_rate(prop_eligible: pd.Series) -> np.ndarray:
        """Determine the hold harmless rate for a district based on the
        proportion of eligible children.
        """
        return np.where(
            prop_eligible < 0.15,
            0.85,
            np.where(
                prop_eligible < 0.3,
                0.9,
                0.95
            )
        )

    @staticmethod
    def _state_minimum_global(grant_type: str) -> int:
        # drawing this manually from the FY 2021 state-level data
        # NOTE: only works for 2021 data
        if grant_type == "basic":
            return 17744098
        if grant_type == "concentration":
            return 3378479
        if grant_type == "targeted":
            return 15083659
        raise ValueError(f"Unmatched grant type: {grant_type}")
