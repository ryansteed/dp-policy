from dp_policy.titlei.utils import weighting, get_allocation_data

import numpy as np
import pandas as pd
import scipy.stats as stats


class Allocator:
    def __init__(
        self,
        estimates,
        congress_cap=0.4,
        adj_sppe_bounds=[0.32, 0.48],
        adj_sppe_bounds_efig=[0.34, 0.46],
        verbose=False
    ):
        self.estimates = estimates
        self.congress_cap = congress_cap
        self.adj_sppe_bounds = adj_sppe_bounds
        self.adj_sppe_bounds_efig = adj_sppe_bounds_efig
        self.verbose = verbose

    def allocations(
        self, uncertainty=True, **uncertainty_params
    ) -> pd.DataFrame:
        self.calc_auth()
        if uncertainty:
            self.calc_uncertainty(**uncertainty_params)
        return self.estimates

    def calc_auth(self):
        """
        Appends the allocated grants as columns to the estimates DataFrame.

        Must generate at least `true_grant_total` and `est_grant_total`.

        returns:
            pd.DataFrame: current estimates
        """
        raise NotImplementedError

    def calc_uncertainty(self):
        """Calculate upper and lower confidence bounds.
        NOTE: doesn't account for hold harmless provision.
        """
        raise NotImplementedError

    def normalize(self):
        """Normalize authorization amounts to allocation amounts.
        """

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


class Authorizer(Allocator):
    def grant_types(self):
        raise NotImplementedError

    def allocations(
        self, uncertainty=True, normalize=True, **uncertainty_params
    ) -> pd.DataFrame:
        super().allocations(uncertainty=uncertainty, **uncertainty_params)
        if normalize:
            self._normalize(uncertainty=uncertainty)
        return self.estimates

    def _normalize(self, uncertainty=True):
        # get this year's budget
        true_allocs = get_allocation_data("../data/titlei-allocations_20")
        actual_budget = true_allocs["HistAlloc"].sum()
        for prefix in ("true", "est"):
            current_budget = self.estimates[f"{prefix}_grant_total"].sum()
            authorization_amounts = [
                f"{prefix}_grant_{kind}"
                for kind in self.grant_types()
            ]
            if uncertainty:
                authorization_amounts += [
                    f"{prefix}_grant_{kind}_{suffix}"
                    for kind in (t for t in self.grant_types() if t != "total")
                    for suffix in ("ci_upper", "ci_lower")
                ] + [f"{prefix}_ci_upper", f"{prefix}_ci_lower"]
            self.estimates.loc[:, authorization_amounts] = \
                self.estimates[authorization_amounts].apply(
                    lambda x: Authorizer.normalize_to_budget(
                        x, actual_budget
                    )
                )
            if self.verbose:
                print(
                    f"{current_budget} authorized reduced "
                    f"to {actual_budget} allocated."
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


class SonnenbergAuthorizer(Authorizer):
    def __init__(self, *args, hold_harmless=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_harmless = hold_harmless

    def allocations(
        self, uncertainty=True, **uncertainty_params
    ) -> pd.DataFrame:
        super().allocations(uncertainty=uncertainty, **uncertainty_params)
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

        prefixes = ("true", "est")
        # calculate grant amounts for true/randomized values
        for prefix in prefixes:
            # BASIC GRANTS
            # authorization calculation
            self.estimates[f"{prefix}_grant_basic"] = \
                self.estimates[f"{prefix}_children_eligible"] * adj_sppe
            # For basic grants, LEA must have >10 eligible children
            # AND >2% eligible
            count_eligible = \
                self.estimates[f"{prefix}_children_eligible"] > 10
            share_eligible = (
                self.estimates[f"{prefix}_children_eligible"]
                / self.estimates[f"{prefix}_children_total"]
            ) > 0.02
            self.estimates[f"{prefix}_eligible_basic"] = \
                count_eligible & share_eligible
            self.estimates.loc[
                ~self.estimates[f"{prefix}_eligible_basic"],
                f"{prefix}_grant_basic"
            ] = 0.0

            # CONCENTRATION GRANTS
            # For concentration grants, LEAs must meet basic eligibility
            # AND have either
            # a) >6500 eligible
            # b) 15% of pop. is eligible
            self.estimates[f"{prefix}_grant_concentration"] = \
                self.estimates[f"{prefix}_grant_basic"]
            count_eligible = \
                self.estimates[f"{prefix}_children_eligible"] > 6500
            prop = self.estimates[f"{prefix}_children_eligible"] \
                / self.estimates[f"{prefix}_children_total"]
            prop_eligible = prop > 0.15
            self.estimates[f"{prefix}_eligible_concentration"] = \
                count_eligible | prop_eligible
            self.estimates.loc[
                ~self.estimates[f"{prefix}_eligible_concentration"],
                f"{prefix}_grant_concentration"
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
                self.estimates[f"{prefix}_children_eligible"] > 10
            share_eligible = self.estimates[f"{prefix}_children_eligible"] \
                / self.estimates[f"{prefix}_children_total"]
            prop_eligible = share_eligible > 0.05
            self.estimates[f"{prefix}_eligible_targeted"] = \
                count_eligible & prop_eligible
            self.estimates.loc[
                ~self.estimates[f"{prefix}_eligible_targeted"],
                f"{prefix}_grant_targeted"
            ] = 0.0

            # EFIG
            # TODO

            # clip lower bound to zero
            for grant_type in ["basic", "concentration", "targeted"]:
                self.estimates.loc[
                    self.estimates[f"{prefix}_grant_{grant_type}"] < 0.0,
                    f"{prefix}_grant_{grant_type}"
                ] = 0.0

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
            self.calc_bounds(prefix, cv_z, e_nu)

        self.calc_expected_loss()

        return self.estimates.true_ci_lower, self.estimates.true_ci_upper

    def calc_bounds(self, prefix, cv_z, e_nu):
        mu_hat, _, nu_hat, k = self.get_vars(prefix)

        # see Notion for notes on how to calculate these
        # basic grants
        self.estimates[f"{prefix}_grant_basic_ci_upper"] = \
            k * mu_hat * (1 + cv_z)
        self.estimates.loc[
            mu_hat < np.maximum(10, 0.02*nu_hat*(1-e_nu)) / (1+cv_z),
            f"{prefix}_grant_basic_ci_upper"
        ] = 0.0
        self.estimates[f"{prefix}_grant_basic_ci_lower"] = \
            k * mu_hat * (1 - cv_z)
        self.estimates.loc[
            mu_hat < np.maximum(10, 0.02*nu_hat*(1+e_nu)) / (1-cv_z),
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
            # clip lower bound to zero
            for grant_type in ["basic", "concentration", "targeted"]:
                self.estimates.loc[
                    self.estimates[
                        f"{prefix}_grant_{grant_type}_ci_{bound}"
                    ] < 0.0,
                    f"{prefix}_grant_{grant_type}_ci_{bound}"
                ] = 0.0

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
            assert (self.estimates[
                f"{prefix}_grant_{grant_type}_ci_upper"
            ] < self.estimates[
                f"{prefix}_grant_{grant_type}"
            ]).sum() == 0
            assert (self.estimates[
                f"{prefix}_grant_{grant_type}_ci_lower"
            ] > self.estimates[
                f"{prefix}_grant_{grant_type}"
            ]).sum() == 0

    def calc_prob_eligibility(self):
        mu_hat, sigma_mu, nu_hat, _ = self.get_vars("true")

        # basic
        self.estimates["prob_el_basic"] = \
            SonnenbergAuthorizer._calc_prob(
                nu_hat, mu_hat, sigma_mu, 10, 0.02
            )
        # concentration
        self.estimates["prob_el_concentration"] = \
            SonnenbergAuthorizer._calc_prob(
                nu_hat, mu_hat, sigma_mu, 6500, 0.15, union=True
            )
        # targeted
        self.estimates["prob_el_targeted"] = \
            SonnenbergAuthorizer._calc_prob(
                nu_hat, mu_hat, sigma_mu, 10, 0.05
            )

    @staticmethod
    def _calc_prob(
        nu_hat, mu_hat, sigma_mu,
        raw: float, prop: float, union: bool = False
    ):
        """Calculate the likelihood of eligibility when there are two required
        conditions.

        Args:
            raw_ub (float): lower bound on poverty count
            prop_ub (float): lower bound on poverty share
            union (bool): if the eligibility requirement is a union
        """
        # assumes e_nu = 0
        return np.where(
            (nu_hat > raw / prop) if union else (nu_hat <= raw / prop),
            1 - stats.norm.cdf((raw-mu_hat)/sigma_mu),
            1 - stats.norm.cdf((prop*nu_hat - mu_hat)/sigma_mu)
        )  # nu is point mass

    def calc_expected_loss(self):
        # assumes e_nu = 0
        self.calc_prob_eligibility()

        # basic
        mu_hat, _, _, k = self.get_vars("true")
        expected_y = k*mu_hat*self.estimates["prob_el_basic"]
        y_hat = self.estimates.true_grant_basic
        self.estimates["expected_loss_basic"] = expected_y - y_hat
        return self.estimates.expected_loss_basic

    def get_vars(self, prefix):
        mu_hat = self.estimates[f"{prefix}_children_eligible"]
        sigma_mu = self.estimates.cv * mu_hat
        nu_hat = self.estimates[f"{prefix}_children_total"]
        k, _ = self.adj_sppe()
        return mu_hat, sigma_mu, nu_hat, k

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

    def _hold_harmless(self):
        print("Holding harmless")
        # load last year's allocs - watch out for endogeneity
        # get this year's budget
        true_allocs = get_allocation_data("../data/titlei-allocations_19")\
            .rename(columns={
                'HistAlloc': 'alloc_2019'
            })
        self.estimates = self.estimates.join(
            true_allocs["alloc_2019"]
        )
        for prefix in ("true", "est"):
            for grant_type in ["basic", "concentration", "targeted"]:
                pass
        # limit losses to 15%

        raise NotImplementedError
