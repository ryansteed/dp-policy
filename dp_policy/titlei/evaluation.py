import os
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
import pickle

from dp_policy.titlei.utils import get_acs_unified
import dp_policy.config as config

from typing import Dict, Callable


def get_geography():
    """Load shapefiles for LEAs.
    """
    geo = gpd.read_file(os.path.join(
        config.root,
        "data/shapefiles/school_districts_19/schooldistrict_sy1819_tl19.shp"
    ))
    geo.STATEFP = geo.STATEFP.astype(int)
    geo["District ID"] = np.where(
        geo.UNSDLEA.notna(),
        geo.UNSDLEA,
        np.where(
            geo.SCSDLEA.notna(),
            geo.SCSDLEA,
            geo.ELSDLEA
        )
    )
    geo["District ID"] = geo["District ID"].astype(int)
    geo = geo.rename(columns={
        "STATEFP": "State FIPS Code"
    }).set_index(["State FIPS Code", "District ID"])
    return geo


def discrimination_join(
    results: pd.DataFrame,
    save_path: str = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Join results to demographic covariates.

    Args:
        results (pd.DataFrame): Results, keyed by LEA.
        save_path (str, optional): Where to save the joined dataframe.
            Defaults to None.
        verbose (bool, optional): Defaults to False.

    Returns:
        pd.DataFrame: Joined dataframe.
    """
    acs = get_acs_unified(verbose)

    variables = [
        'Total population (RACE) - est',
    ]
    # add race variables
    variables += [
        r for r in acs.columns
        if r.endswith("(RACE) - pct")
        and "and" not in r
        and "races" not in r
        and not r.startswith("One race")
    ] + ["Two or more races (RACE) - pct"]
    # add ethnicity variables
    hisp = [
        r for r in acs.columns
        if r.endswith("(HISPANIC OR LATINO AND RACE) - pct")
    ]
    variables += hisp[1:6]
    # add income variables
    variables += [
        r for r in acs.columns
        if r.startswith("Median household income (dollars) (")
    ]
    # add rural/urban - need a 3rd data source
    # add immigrant status
    variables += [
        # "Foreign born (PLACE OF BIRTH) - est",
        "Foreign born (PLACE OF BIRTH) - pct",
        # "Not a U.S. citizen (U.S. CITIZENSHIP STATUS) - est",
        "Not a U.S. citizen (U.S. CITIZENSHIP STATUS) - pct"
    ]
    # add language isolation
    variables += [
        # 'Language other than English (LANGUAGE SPOKEN AT HOME) - est',
        'Language other than English (LANGUAGE SPOKEN AT HOME) - pct'
    ]
    # add renters vs. homeowners (housing security)
    variables += [
        # 'Renter-occupied (HOUSING TENURE) - est',
        'Renter-occupied (HOUSING TENURE) - pct',
        'Average household size of renter-occupied unit (HOUSING TENURE) - est'
    ]
    # look in these columns for '-' and replace with nan according to ACS docs
    # https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-estimate-and-annotation-values.html
    # otherwise convert to numeric
    acs_vars = acs[variables]\
        .replace(['-', "**", "***", "(X)", "N", "null"], np.nan)\
        .replace('250,000+', 250000)\
        .apply(pd.to_numeric, errors='raise')

    if verbose:
        print(variables)
        print(acs_vars.shape)
        print(results.shape)

    # adding geographic area
    geo = get_geography()

    to_join = acs_vars.join(geo["ALAND"], how="inner")
    if verbose:
        print("ACS", acs_vars.shape)
        print("Geo joined ACS", to_join.shape)
        print(
            to_join[
                to_join["Total population (RACE) - est"].isna()
            ].groupby(["State FIPS Code", "District ID"]).groups.keys()
        )

    print("Joining ACS/geo variables...")
    grants = results.join(to_join, how="inner")
    if verbose:
        print(
            "missing some districts in ACS/geo:",
            len(results.groupby([
                "State FIPS Code", "District ID"
            ])) - len(grants.groupby([
                "State FIPS Code", "District ID"
            ]))
        )
        print(grants.shape)

    if save_path:
        print("Saving to feather...")
        grants.reset_index().to_feather(f"{save_path}.feather")
        print("... saved.")
    return grants


def discrimination_treatments_join(
    treatments_name: dict,
    exclude: list = [],
    epsilon: float = 0.1,
    delta: float = 0.0
) -> pd.DataFrame:
    """Join results to demographic covariates.

    Args:
        treatments_name (dict): Treatment results keyed by treatment name.
        exclude (list, optional): Treatments to exclude. Defaults to [].
        epsilon (float, optional): Epsilon to use (one allowed). Defaults to
            0.1.
        delta (float, optional): Delta to use (one allowed). Defaults to 0.0.

    Returns:
        _type_: Combined dataframe for all treatments.
    """
    # output a concatenated DF with a new index column indicating which
    # treatment was applied
    treatments = {
        treatment: df.loc[(
            slice(None),
            delta if delta is not None else slice(None),
            epsilon if epsilon is not None else slice(None),
            slice(None),
            slice(None)
        ), [
            c for c in df.columns
            if c.endswith("- pct")
            or c.endswith("- est")
            or c.startswith("true")
            or c.endswith("grant_total")
            or c in [
                "ALAND"
            ]
        ]]
        for treatment, df in load_treatments(treatments_name).items()
        if treatment not in exclude
    }
    print("Saving", treatments.keys())
    joined = pd.concat(
        treatments,
        names=['treatment']
    )
    print("Concatenated has {} rows and {} treatments".format(
        len(joined),
        len(treatments)
    ))
    discrimination_joined = discrimination_join(
        joined,
        save_path=f"{config.root}/results/policy_experiments/"
        f"{treatments_name}_discrimination_laplace"
    )
    return discrimination_joined


def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{}'.format(n*100)
    return percentile_


def geo_join(results: pd.DataFrame) -> pd.DataFrame:
    """Join results wwith shapefiles.

    Args:
        results (pd.DataFrame): Results, keyed by LEA.

    Returns:
        pd.DataFrame: Results joined with LEA shapefile and geographic data.
    """
    # NOTE: only accepts "State FIPS Code", "District ID", "trial" in index
    results = results.copy()
    results["error"] = results.est_grant_total - results.true_grant_total
    results["error_per_child"] = results.error / results['true_children_total']
    results["error_per_child_eligible"] = \
        results.error / results['true_children_eligible']
    results["error_dp"] = results.dpest_grant_total - results.true_grant_total
    results["error_dp_per_child"] = \
        results.error_dp / results['true_children_total']
    results["error_dp_per_child_eligible"] = \
        results.error_dp / results['true_children_eligible']

    results["percent_eligible"] = \
        results["true_children_eligible"] / results["true_children_total"]
    results["switched_eligibility"] = \
        ~(
            (results.est_eligible_targeted == results.true_eligible_targeted)
            & (results.est_eligible_basic == results.true_eligible_basic)
            & (
                results.est_eligible_concentration ==
                results.true_eligible_concentration
            )
        )
    results["became_eligible"] = \
        (
            results.est_eligible_targeted.astype(bool)
            & ~results.true_eligible_targeted.astype(bool)
        ) \
        | (
            results.est_eligible_basic.astype(bool)
            & ~results.true_eligible_basic.astype(bool)
        ) \
        | (
            results.est_eligible_concentration.astype(bool)
            & ~results.true_eligible_concentration.astype(bool)
        )
    results["became_ineligible"] = \
        (
            ~results.est_eligible_targeted.astype(bool)
            & results.true_eligible_targeted.astype(bool)
        ) \
        | (
            ~results.est_eligible_basic.astype(bool)
            & results.true_eligible_basic.astype(bool)
        ) \
        | (
            ~results.est_eligible_concentration.astype(bool)
            & results.true_eligible_concentration.astype(bool)
        )
    results["switched_eligibility_dp"] = \
        ~(
            (results.dpest_eligible_targeted == results.true_eligible_targeted)
            & (results.dpest_eligible_basic == results.true_eligible_basic)
            & (
                results.dpest_eligible_concentration ==
                results.true_eligible_concentration
            )
        )
    results["became_eligible_dp"] = \
        (
            results.dpest_eligible_targeted.astype(bool)
            & ~results.true_eligible_targeted.astype(bool)
        ) \
        | (
            results.dpest_eligible_basic.astype(bool)
            & ~results.true_eligible_basic.astype(bool)
        ) \
        | (
            results.dpest_eligible_concentration.astype(bool)
            & ~results.true_eligible_concentration.astype(bool)
        )
    results["became_ineligible_dp"] = \
        (
            ~results.dpest_eligible_targeted.astype(bool)
            & results.true_eligible_targeted.astype(bool)
        ) \
        | (
            ~results.dpest_eligible_basic.astype(bool)
            & results.true_eligible_basic.astype(bool)
        ) \
        | (
            ~results.dpest_eligible_concentration.astype(bool)
            & results.true_eligible_concentration.astype(bool)
        )
    results["dp_marginal"] = \
        results["error_dp_per_child_eligible"] -\
        results["error_per_child_eligible"]

    geo = get_geography()
    joined = geo.join(
        results[[
            "error",
            "error_per_child",
            "error_per_child_eligible",
            "error_dp",
            "error_dp_per_child",
            "error_dp_per_child_eligible",
            "true_children_eligible",
            "true_pop_total",
            "percent_eligible",
            "true_grant_total",
            "switched_eligibility",
            "became_eligible",
            "became_ineligible",
            "switched_eligibility_dp",
            "became_eligible_dp",
            "became_ineligible_dp",
            "dp_marginal"
        ]]
        .groupby(["State FIPS Code", "District ID"])
        .agg(['mean', 'std', 'sem', percentile(0.05)]),
        how="inner"
    )
    joined.columns = [
        col if isinstance(col, str) else '_'.join([
            c for c in col if c != 'mean'
        ]).rstrip('_')
        for col in joined.columns.values
    ]
    for col in [
        "error_per_child",
        "error_per_child_eligible",
        "error_dp_per_child",
        "error_dp_per_child_eligible"
    ]:
        joined.loc[
            np.isinf(joined[col]), col
        ] = np.nan

    return joined


def plot_treatments(
    treatments: Dict[str, pd.DataFrame],
    x_func: Callable,
    plot_method: Callable,
    plot_kwargs: dict,
    filename: str = None,
    xlab: str = None,
    ylab: str = "Smoothed density",
    grant: str = "total",
    epsilon: float = None,
    delta: float = None,
    mean_line: bool = False
):
    """Plot treatment results.

    Args:
        treatments (Dict[str, pd.DataFrame]): Treatment results mapped by
            treatment name.
        x_func (Callable): Function to transform dataframe before plotting.
        plot_method (Callable): Function for plotting.
        plot_kwargs (dict): Parameters for plotting function.
        filename (str, optional): Filename of plot. Defaults to None.
        xlab (str, optional): X label for plot. Defaults to None.
        ylab (str, optional): Y label for plot. Defaults to "Smoothed density".
        grant (str, optional): Grant type to plot. Defaults to "total".
        epsilon (float, optional): Epsilon to plot. Defaults to None.
        delta (float, optional): Delta to plot. Defaults to None.
        mean_line (bool, optional): Whether to include mean line. Defaults to
            False.
    """
    palette = sns.color_palette(n_colors=len(treatments))
    for i, (treatment, df_raw) in enumerate(treatments.items()):
        df = df_raw.loc[pd.IndexSlice[
            :,
            delta if delta is not None else slice(None),
            epsilon if epsilon is not None else slice(None),
            :,
            :
        ], :].copy()

        df.loc[:, "misalloc"] = \
            df[f"dpest_grant_{grant}"] - df[f"true_grant_{grant}"]
        df.loc[:, "misalloc_sq"] = np.power(df["misalloc"], 2)
        if grant == "total":
            df["lost_eligibility"] = \
                (
                    df["dpest_eligible_basic"].astype(bool) &
                    ~df["true_eligible_basic"].astype(bool)
                ) |\
                (
                    df["dpest_eligible_concentration"].astype(bool) &
                    ~df["true_eligible_concentration"].astype(bool)
                ) |\
                (
                    df["dpest_eligible_targeted"].astype(bool) &
                    ~df["true_eligible_targeted"].astype(bool)
                )
        else:
            df["lost_eligibility"] = \
                ~df["dpest_eligible_{}".format(grant)].astype(bool) \
                & df["true_eligible_{}".format(grant)].astype(bool)
        plot_kwargs.update({
            'label': treatment,
            'color': palette[i]
        })
        x = x_func(df)
        plot_method(x, **plot_kwargs)
        if mean_line:
            plt.axvline(
                x.mean(), color=palette[i],
                linestyle='dashed'
            )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='upper right')
    if filename:
        plt.savefig(
            f"{config.root}/plots/bootstrap/{filename}.png",
            dpi=100,
            bbox_inches='tight'
        )
    plt.show()
    plt.close()


def cube(x):
    return np.sign(x)*np.power(np.abs(x), 1/3)


def heatmap(
    data: pd.DataFrame,
    label: str = None,
    title: str = None,
    transform: str = 'cube',
    theme: str = "seismic_r",
    y: str = "error_dp_per_child",
    vcenter: int = 0,
    file: str = None,
    figsize: tuple = (10, 5),
    bar_location: str = 'bottom',
    min: int = None,
    max: int = None,
    dpi: int = 300,
    alpha: float = 0.1
):
    """Plot heatmap of results.

    Args:
        data (pd.DataFrame): Dataframe to plot from.
        label (str, optional): Label for colorbar. Defaults to None.
        title (str, optional): Title for plot. Defaults to None.
        transform (str, optional): Transformation for data. Defaults to 'cube'.
        theme (str, optional): Theme for colorbar. Defaults to "seismic_r".
        y (str, optional): Column to plot. Defaults to "error_dp_per_child".
        vcenter (int, optional): Center of colorbar. Defaults to 0.
        file (str, optional): Filename. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (10, 5).
        bar_location (str, optional): Where to place the colorbar. Defaults to
            'bottom'.
        min (int, optional): Minimum for colorbar. Defaults to None.
        max (int, optional): Maximum for colorbar. Defaults to None.
        dpi (int, optional): Figure DPI. Defaults to 300.
        alpha (float, optional): Confidence level for t-test. Defaults to 0.1.
    """
    if alpha is not None:
        data[f"{y}_moe"] = data.loc[:, f"{y}_sem"] * stats.norm.ppf(1 - alpha / 2)
        sig = ~(
            ((data[y] + data[f"{y}_moe"]) >= 0) &
            ((data[y] - data[f"{y}_moe"]) <= 0)
        )
        print(
            "All but",
            len(data)-sig.sum(),
            f"are significantly different from zero at {alpha}"
        )

    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)

    for key in [y, f"{y}_moe"] if alpha is not None else [y]:
        if transform == 'cube':
            data.loc[:, key] = cube(data[key])
        if transform == 'log':
            data.loc[:, key] = np.where(data[key] == 0, 0, np.log(data[key]))
        if transform == 'sqrt':
            data.loc[:, key] = np.sign(data[key]) * np.sqrt(np.abs(data[key]))

    # Create colorbar as a legend
    if min is None:
        min = data[y].min()
    if max is None:
        max = data[y].max()

    bound = np.max(np.abs([min, max]))

    if vcenter is not None and transform != 'log':
        norm = pltc.TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound)
    else:
        norm = pltc.Normalize(vmin=min, vmax=max)
    sm = plt.cm.ScalarMappable(cmap=theme, norm=norm)

    if alpha is not None:
        print(
            f"None of the {(1-alpha)*100}% MOEs exceeds",
            data[f"{y}_moe"].abs().max()
        )
        data[f"{y}_sig"] = np.where(sig, data[y], np.nan)
    data.plot(
        column=f"{y}_sig" if alpha is not None else y,
        cmap=theme,
        norm=norm,
        ax=ax,
        linewidth=0.05,
        edgecolor='0.1',
        missing_kwds=dict(
            hatch='///',
            edgecolor=(0, 0, 0, 0.25),
            facecolor='none',
            label=f"Not significant (p < {alpha})"
        )
    )
    cb = fig.colorbar(
        sm,
        location=bar_location,
        shrink=0.5, pad=0.05, aspect=30
    )
    cb.set_label(label)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if file is not None:
        plt.savefig(
            f"{config.root}/plots/geo/{file}",
            dpi=dpi, bbox_inches='tight'
        )
        plt.close()
    else:
        plt.show()


def save_treatments(treatments: Dict[str, pd.DataFrame], experiment_name: str):
    # minify treatments
    treatments = {
        treatment: df.loc[:, [
            c for c in df.columns
            if c.endswith("- pct")
            or c.endswith("- est")
            or c.startswith("true")
            or "eligible" in c
            or "grant" in c
            or c in [
                "ALAND"
            ]
        ]]
        for treatment, df in treatments.items()
    }
    pickle.dump(
        treatments,
        open(
            f"{config.root}/results/policy_experiments/{experiment_name}.pkl",
            'wb'
        )
    )


def load_treatments(experiment_name) -> Dict[str, pd.DataFrame]:
    return pickle.load(
        open(
            f"{config.root}/results/policy_experiments/{experiment_name}.pkl",
            'rb'
        )
    )


def compare_treatments(
  treatments: Dict[str, pd.DataFrame],
  epsilon: float = 0.1,
  delta: float = 0.0,
  mapvar: str = "error_per_child",
  experiment_name: str = None
):
    """Compare treatment results.

    Args:
        treatments (Dict[str, pd.DataFrame]): Treatment results mapped by
            treatment names.
        epsilon (float, optional): Epsilon to use. Defaults to 0.1.
        delta (float, optional): Delta to use. Defaults to 0.0.
        mapvar (str, optional): Which variable to plot. Defaults to
            "error_per_child".
        experiment_name (str, optional): Name of the experiment. Defaults to
            None.
    """
    if epsilon is None:
        print(
            "[WARN] Epsilon is none "
            "- only use this if there is only one eps value in the df."
        )
    else:
        print("Comparing at eps=", epsilon)
    for treatment, df in treatments.items():
        df = df.loc[pd.IndexSlice[
            :,
            delta if delta is not None else slice(None),
            epsilon if epsilon is not None else slice(None),
            :,
            :
        ], :].copy()
        treatments[treatment] = df
        print(len(df))
        print("\n#", treatment)
        print("True budget:", df[f"true_grant_total"].sum())
        print("DP est budget:",  df[f"dpest_grant_total"].sum())
        df["became_ineligible"] = \
            (
                ~df.dpest_eligible_targeted.astype(bool)
                & df.true_eligible_targeted.astype(bool)
            )\
            | (
                ~df.dpest_eligible_basic.astype(bool)
                & df.true_eligible_basic.astype(bool)
            )\
            | (
                ~df.dpest_eligible_concentration.astype(bool)
                & df.true_eligible_concentration.astype(bool)
            )
        print(
            "Avg prop. districts erroneously ineligible:",
            df.became_ineligible.groupby("trial").sum().mean()
        )

        print("# est")
        misalloc_statistics(
            df.est_grant_total - df.true_grant_total
        )

        print("# dpest")
        misalloc_statistics(
            df.dpest_grant_total - df.true_grant_total
        )

        print("# marginal")
        misalloc_statistics(
            df.dpest_grant_total - df.est_grant_total
        )

    # compare bias
    plot_treatments(
        treatments,
        lambda df: np.sqrt(df.groupby('trial')["misalloc"].mean()),
        sns.kdeplot,
        dict(bw_method=0.5, fill=True),
        epsilon=epsilon,
        delta=delta,
        xlab=f"Mean misalloc (per trial)",
        mean_line=True
    )

    # compare RMSE - mean and hist
    plot_treatments(
        treatments,
        lambda df: np.sqrt(df.groupby('trial')["misalloc_sq"].mean()),
        sns.kdeplot,
        dict(bw_method=0.5, fill=True),
        filename=f"{experiment_name}_rmse",
        epsilon=epsilon,
        delta=delta,
        xlab=f"RMSE (per trial)",
        mean_line=True
    )

    # compare likelihood of ineligibility
    plot_treatments(
        treatments,
        lambda df:
            df.groupby(['State FIPS Code', 'District ID'])["lost_eligibility"]
            .mean(),
        sns.kdeplot,
        dict(bw_method=0.5, fill=True),
        # filename=f"likelihood_ineligible_total",
        xlab=f"Likelihood of losing eligibility",
        epsilon=epsilon,
        delta=delta,
        mean_line=True
    )

    # compare nationwide map
    print("Plotting", mapvar)
    ymins = []
    ymaxs = []
    treatments_geo = {
        treatment:
            geo_join(
                df.loc[pd.IndexSlice[
                    :,
                    delta if delta is not None else slice(None),
                    epsilon if epsilon is not None else slice(None),
                    :,
                    :
                ]]
            )
        for treatment, df in treatments.items()
    }
    for treatment, df in treatments_geo.items():
        err = cube(df.loc[[
            f for f in df.index.get_level_values("State FIPS Code").unique()
            if f not in [2, 15]
        ]]["error_per_child"])
        ymins.append(err.min())
        ymaxs.append(err.max())
    ymin = np.min(ymins)
    ymax = np.max(ymaxs)
    for treatment, df in treatments_geo.items():
        heatmap(
            df.loc[[
                f
                for f in df.index.get_level_values("State FIPS Code").unique()
                if f not in [2, 15]
            ]],
            y="error_per_child_eligible",
            label="Misallocation per eligible child (cube root)",
            title=treatment,
            file=f"{experiment_name}_{treatment}.png",
            figsize=(15, 10),
            bar_location='right',
            min=ymin,
            max=ymax,
            dpi=50
        )


def match_true(
    df_true: pd.DataFrame,
    dfs_to_match: pd.DataFrame
):
    """Equalize result baselines (columns with "true").

    Args:
        df_true (pd.DataFrame): Dataframe with ground truth.
        dfs_to_match (pd.DataFrame): Dataframes to equalize to `df_true`.
    """
    for c in (c for c in df_true.columns if "true" in c):
        for df in dfs_to_match:
            df.loc[:, c] = df_true.loc[:, c]


def misalloc_statistics(
    error: pd.Series,
    allocations: pd.DataFrame = None,
    grant_type: str = None
):
    err_grouped = error.groupby(
        ["State FIPS Code", "District ID"]
    )
    exp_error = err_grouped.mean()
    print(f"# rows: {len(error)}")
    print(f"Max error: {np.abs(error).max()}")

    print("-- RMSE --")
    print(f"RMSE:", np.sqrt(np.mean(error**2)))
    print(
        "Avg. RMSE",
        np.mean(
            np.sqrt((error**2).groupby('trial').mean())
        )
    )
    print(
        f"RMSE in exp. error:",
        np.sqrt(np.mean(exp_error**2))
    )

    print("-- Losses --")
    print(
        "Avg. (per trial) # of districts losing $$:",
        (error < 0).groupby("trial").sum().mean()
    )
    print(
        f"Avg. total losses:",
        error[
            error < 0
        ].abs().groupby("trial").sum().mean()
    )
    print(
        f"Std. total losses:",
        error[
            error < 0
        ].abs().groupby("trial").sum().std()
    )
    print(
        "Total exp losses:",
        exp_error[exp_error < 0].abs().sum()
    )
    print(
        "Average exp loss",
        exp_error[exp_error < 0].abs().mean()
    )
    lowerr = error.groupby(["State FIPS Code", "District ID"]).quantile(0.05)
    print(
        "Total 5% quantile losses:",
        lowerr[lowerr < 0].abs().sum()
    )
    print(
        "Avg. 5% quantile loss:",
        lowerr[lowerr < 0].mean()
    )

    print("-- Misalloc --")
    print(
        f"Avg. total abs misalloc:",
        error.abs().groupby("trial").sum().mean()
    )
    print(
        f"Total exp abs misalloc:",
        exp_error.abs().sum()
    )

    if allocations is not None:
        print("-- Other stats --")
        small_district = allocations["true_pop_total"]\
            .groupby(["State FIPS Code", "District ID"])\
            .first() < 20000
        print(
            "# small districts:",
            small_district.sum()
        )
        print(
            "Total exp misalloc to large districts:",
            exp_error[~small_district].abs().sum()
        )
        print(
            "Total exp misalloc to small districts:",
            exp_error[small_district].abs().sum()
        )

        if grant_type is not None:
            print(
                "Total true alloc:",
                allocations[f"true_grant_{grant_type}"]
                .groupby(["State FIPS Code", "District ID"])
                .first().abs().sum()
            )
            print(
                "Total true alloc per child eligible",
                allocations[f"true_grant_{grant_type}"]
                .groupby(["State FIPS Code", "District ID"])
                .first().sum() / allocations[f"true_children_eligible"]
                .groupby(["State FIPS Code", "District ID"])
                .first().sum()
            )
            print("Average true alloc: {}".format(
                allocations[f"true_grant_{grant_type}"].mean()
            ))
            print(
                "Average true alloc per child eligible",
                (
                    allocations[f"true_grant_{grant_type}"] /
                    allocations["true_children_eligible"]
                ).mean()
            )
            print("Max true alloc: {}".format(
                allocations[f"true_grant_{grant_type}"].max()
            ))
