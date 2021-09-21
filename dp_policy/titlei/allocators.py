from dp_policy.titlei.utils import weighting

import numpy as np
import pandas as pd
import scipy.stats as stats


class Allocator:
    def __init__(
        self,
        estimates,
        congress_cap=0.4,
        adj_sppe_bounds=[0.32, 0.48],
        adj_sppe_bounds_efig=[0.34, 0.46]
    ):
        self.estimates = estimates
        self.congress_cap = congress_cap
        self.adj_sppe_bounds = adj_sppe_bounds
        self.adj_sppe_bounds_efig = adj_sppe_bounds_efig

    def allocations(self, **uncertainty_params) -> pd.DataFrame:
        self.calc_alloc()
        self.calc_uncertainty(**uncertainty_params)
        return self.estimates

    def calc_alloc(self):
        """
        Appends the allocated grants as columns to the estimates DataFrame.

        Must generate at least `true_grant_total` and `est_grant_total`.

        returns:
            pd.DataFrame: current estimates
        """
        raise NotImplementedError

    def calc_uncertainty(self):
        """Calculate upper and lower confidence bounds.
        """
        raise NotImplementedError

    def adj_sppe(self):
        avg_sppe = np.mean(self.estimates.sppe)
        adj_sppe = self.estimates.sppe * self.congress_cap
        adj_sppe_trunc = adj_sppe.clip(
            *np.array(self.adj_sppe_bounds)*avg_sppe
        )
        adj_sppe_efig = adj_sppe.clip(
            *np.array(self.adj_sppe_bounds_efig)*avg_sppe
        )
        return adj_sppe_trunc, adj_sppe_efig


class AbowdAllocator(Allocator):
    """
    As described in https://arxiv.org/pdf/1808.06303.pdf
    """
    def calc_alloc(self):
        adj_sppe, _ = self.adj_sppe()

        self.estimates["adj_sppe"] = adj_sppe
        self.estimates["true_grant_total"] = \
            adj_sppe * self.estimates.true_children_eligible
        self.estimates["est_grant_total"] = \
            adj_sppe * self.estimates.est_children_eligible
        return self.estimates

    def calc_uncertainty(self, alpha=0.05):
        z = stats.norm.ppf(1-alpha/2)
        self.estimates["true_se"] = \
            self.estimates.true_grant_total * self.estimates.cv
        self.estimates["true_ci_lower"] = \
            self.estimates.true_grant_total - self.estimates.true_se * z
        self.estimates["true_ci_upper"] = \
            self.estimates.true_grant_total + self.estimates.true_se * z
        return self.estimates.true_ci_lower, self.estimates.true_ci_upper


class SonnenbergAuthorizer(Allocator):
    def calc_alloc(self):
        # calc adj. SPPE
        adj_sppe, adj_sppe_efig = self.adj_sppe()

        # calculate grant amounts for true/randomized values
        for prefix in ("true", "est"):
            # BASIC GRANTS
            # authorization calculation
            self.estimates[f"{prefix}_grant_basic"] = \
                self.estimates[f"{prefix}_children_eligible"] * adj_sppe
            # For basic grants, LEA must have >10 eligible children
            # AND >2% eligible
            # NOTE: second criteria off for now, for explanation purposes
            # TODO: turn second criteria back on, also adjust CI calc
            count_eligible = \
                self.estimates[f"{prefix}_children_eligible"] >= 10
            # share_eligible = (
            #     self.estimates[f"{prefix}_children_eligible"]
            #     / self.estimates[f"{prefix}_children_total"]
            # ) >= 0.02
            eligible = count_eligible  # & share_eligible
            self.estimates.loc[~eligible, f"{prefix}_grant_basic"] = 0.0

            # CONCENTRATION GRANTS
            # For concentration grants, LEAs must meet basic eligibility
            # AND have either
            # a) >6500 eligible
            # b) 15% of pop. is eligible
            self.estimates[f"{prefix}_grant_concentration"] = \
                self.estimates[f"{prefix}_grant_basic"]
            count_eligible = \
                self.estimates[f"{prefix}_children_eligible"] >= 6500
            prop = self.estimates[f"{prefix}_children_eligible"] \
                / self.estimates[f"{prefix}_children_total"]
            prop_eligible = prop >= 0.15
            eligible = count_eligible | prop_eligible
            self.estimates.loc[
                ~eligible, f"{prefix}_grant_concentration"
            ] = 0.0

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
            count_eligible = \
                self.estimates[f"{prefix}_children_eligible"] >= 10
            share_eligible = self.estimates[f"{prefix}_children_eligible"] \
                / self.estimates[f"{prefix}_children_total"]
            prop_eligible = share_eligible >= 0.05
            eligible = count_eligible & prop_eligible
            self.estimates.loc[~eligible, f"{prefix}_grant_targeted"] = 0.0

            # EFIG
            # TODO

        self.estimates = SonnenbergAuthorizer.calc_total(self.estimates)

    def calc_uncertainty(self, alpha=0.05, e_nu=0.0):
        """Calculate uncertainty in Sonnenberg authorizations.

        Args:
            alpha (float, optional): confidence level. Defaults to 0.05.
            e_nu (float, optional): assumed error in the total child estimate.
                Defaults to 0.0.
        """
        z = stats.norm.ppf(1-alpha/2)
        cv_z = self.estimates.cv * z

        for prefix in ["true", "est"]:
            mu_hat = self.estimates[f"{prefix}_children_eligible"]
            nu_hat = self.estimates[f"{prefix}_children_total"]
            k, _ = self.adj_sppe()
            # see Notion for notes on how to calculate these
            # basic grants
            self.estimates[f"{prefix}_grant_basic_ci_upper"] = \
                k * mu_hat * (1 + cv_z)
            self.estimates.loc[
                mu_hat < 10 / (1 + cv_z),
                f"{prefix}_grant_basic_ci_upper"
            ] = 0.0
            self.estimates[f"{prefix}_grant_basic_ci_lower"] = \
                k * mu_hat * (1 - cv_z)
            self.estimates.loc[
                mu_hat < 10 / (1 - cv_z),
                f"{prefix}_grant_basic_ci_lower"
            ] = 0.0

            # concentration grants
            self.estimates[f"{prefix}_grant_concentration_ci_upper"] = \
                k * mu_hat * (1 + cv_z)
            self.estimates.loc[
                mu_hat < np.minimum(
                    6500 / 1 + cv_z,
                    0.15*nu_hat*(1-e_nu)/(1+cv_z)
                ),
                f"{prefix}_grant_concentration_ci_upper"
            ] = 0.0
            self.estimates[f"{prefix}_grant_concentration_ci_lower"] = \
                k * mu_hat * (1 - cv_z)
            self.estimates.loc[
                mu_hat < np.minimum(
                    6500 / 1 - cv_z,
                    0.15*nu_hat*(1+e_nu)/(1-cv_z)
                ),
                f"{prefix}_grant_concentration_ci_lower"
            ] = 0.0

            # targeted grants
            self.estimates[f"{prefix}_grant_targeted_ci_upper"] = \
                k * np.apply_along_axis(
                    lambda x: weighting(x[0], x[1]), 1,
                    np.column_stack((mu_hat*(1+cv_z), nu_hat*(1-e_nu)))
                )
            self.estimates.loc[
                mu_hat < np.maximum(10, 0.05*nu_hat) / (1 + cv_z),
                f"{prefix}_grant_targeted_ci_upper"
            ] = 0.0
            self.estimates[f"{prefix}_grant_targeted_ci_lower"] = \
                k * np.apply_along_axis(
                    lambda x: weighting(x[0], x[1]), 1,
                    np.column_stack((mu_hat*(1-cv_z), nu_hat*(1+e_nu)))
                )
            self.estimates.loc[
                mu_hat < np.maximum(10, 0.05*nu_hat) / (1 - cv_z),
                f"{prefix}_grant_targeted_ci_lower"
            ] = 0.0

            # totals
            for bound in ("upper", "lower"):
                self.estimates[f"{prefix}_ci_{bound}"] = \
                    self.estimates[f"{prefix}_grant_basic_ci_{bound}"] + \
                    self.estimates[
                        f"{prefix}_grant_concentration_ci_{bound}"
                    ] + \
                    self.estimates[f"{prefix}_grant_targeted_ci_{bound}"]

            # sanity check the bounds are on the right sides
            for grant_type in ["basic", "concentration", "targeted"]:
                overlapping = (
                    (
                        self.estimates[
                            f"{prefix}_grant_{grant_type}_ci_upper"
                        ] - self.estimates[
                            f"{prefix}_grant_{grant_type}_ci_lower"
                        ]
                    ) < 0.0
                ).sum()
                assert overlapping == 0,\
                    f"{overlapping} overlapping bounds for {grant_type}"

        return self.estimates.true_ci_lower, self.estimates.true_ci_upper

    @staticmethod
    def calc_total(results):
        results["true_grant_total"] = \
            results["true_grant_basic"] + \
            results["true_grant_concentration"] + \
            results["true_grant_targeted"]
        results["est_grant_total"] = \
            results["est_grant_basic"] + \
            results["est_grant_concentration"] + \
            results["est_grant_targeted"]
        return results
