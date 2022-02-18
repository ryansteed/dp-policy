import types
import pandas as pd
from dp_policy.titlei.utils import get_acs_unified

import geopandas as gpd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
import pyarrow.feather as feather
import pickle


def get_geography():
    geo = gpd.read_file(
        "../data/shapefiles/school_districts_19/schooldistrict_sy1819_tl19.shp"
    )
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


def discrimination_join(results, save_path=None, verbose=False):
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
        # print(results[results.index.difference(to_join.index)])
        print(grants.shape)

    if save_path:
        print("Saving to feather...")
        grants.reset_index().to_feather(f"{save_path}.feather")
        print("... saved.")
        # grants.to_csv(f"{save_path}.csv")
    return grants


def discrimination_treatments_join(
    treatments_name,
    exclude=[],
    epsilon=0.1,
    delta=0.0
):
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
        save_path="../results/policy_experiments/"
        f"{treatments_name}_discrimination_laplace"
    )
    return discrimination_joined


def geo_join(results):
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
    # geo.join()
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
        .agg(['mean', 'std', 'sem']),
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
    treatments, x_func, plot_method, plot_kwargs,
    filename=None,
    xlab=None,
    ylab="Smoothed density",
    grant="total",
    epsilon=None,
    delta=None,
    mean_line=False
):
    palette = sns.color_palette(n_colors=len(treatments))
    for i, (treatment, df_raw) in enumerate(treatments.items()):
        df = df_raw.loc[pd.IndexSlice[
            :,
            delta if delta is not None else slice(None),
            epsilon if epsilon is not None else slice(None),
            :,
            :
        ], :].copy()
        # print(df.shape)
        # print(np.unique(df.index.get_level_values(level="epsilon")))
        df.loc[:, "misalloc"] = \
            df[f"dpest_grant_{grant}"] - df[f"true_grant_{grant}"]
        df.loc[:, "misalloc_sq"] = np.power(df["misalloc"], 2)
        if grant == "total":
            # try:
            #     print(~df.true_eligible_basic.astype(bool))
            # except:
            #     print("Failed", df.true_eligible_basic)
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
            f"../plots/bootstrap/{filename}.png",
            dpi=100,
            bbox_inches='tight'
        )
    plt.show()
    plt.close()


def cube(x):
    return np.sign(x)*np.power(np.abs(x), 1/3)


def heatmap(
    data, label=None, title=None, transform='cube', theme="seismic_r",
    y="error_dp_per_child", vcenter=0, file=None,
    figsize=(10, 5), bar_location='bottom', min=None, max=None, dpi=300,
    alpha=0.1
):
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

    for key in [y, f"{y}_moe"]:
        if transform == 'cube':
            data.loc[:, key] = cube(data[key])
        if transform == 'log':
            data.loc[:, key] = np.where(data[key] == 0, 0, np.log(data[key]))
        if transform == 'sqrt':
            data.loc[:, key] = np.sign(data[key]) * np.sqrt(np.abs(data[key]))

    # Create colorbar as a legend
    if min is None and max is None:
        min = data[y].min()
        max = data[y].max()

    bound = np.max(np.abs([min, max]))

    if vcenter is not None and transform != 'log':
        norm = pltc.TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound)
    else:
        norm = pltc.Normalize(vmin=min, vmax=max)
    sm = plt.cm.ScalarMappable(cmap=theme, norm=norm)

    print(
        f"None of the {(1-alpha)*100}% MOEs exceeds",
        data[f"{y}_moe"].abs().max()
    )
    data[f"{y}_sig"] = np.where(sig, data[y], np.nan)
    data.plot(
        column=f"{y}_sig",
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
        sm, location=bar_location, shrink=0.5, pad=0.05, aspect=30
    )
    cb.set_label(label)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if file is not None:
        plt.savefig(f"../plots/geo/{file}", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_treatments(treatments, experiment_name):
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
        open(f"../results/policy_experiments/{experiment_name}.pkl", 'wb')
    )


def load_treatments(experiment_name):
    return pickle.load(
        open(f"../results/policy_experiments/{experiment_name}.pkl", 'rb')
    )


def compare_treatments(
  treatments,
  epsilon=0.1, delta=0.0, mapvar="error_per_child",
  experiment_name=None
):
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
        err = df.dpest_grant_total - df.true_grant_total
        experr = err.groupby(["State FIPS Code", "District ID"]).mean()
        lowerr = err.groupby(["State FIPS Code", "District ID"]).quantile(0.05)
        dperr = df.dpest_grant_total - df.est_grant_total
        expdperr = dperr.groupby(["State FIPS Code", "District ID"]).mean()
        esterr = df.est_grant_total - df.true_grant_total
        expesterr = esterr.groupby(["State FIPS Code", "District ID"]).mean()
        print(
            "RMSE:",
            np.sqrt((err ** 2).mean())
        )
        print(
            "Avg. (per trial) # of districts losing $$:",
            (err < 0).groupby("trial").sum().mean()
        )
        print(
            "Total avg. (per sd) losses:",
            experr[experr < 0].abs().sum()
        )
        print(
            "Total 5% quantile losses:",
            lowerr[lowerr < 0].abs().sum()
        )
        print(
            "Total avg. losses (data error):",
            expesterr[expesterr < 0].abs().sum()
        )
        print(
            "Avg. marginal losses (DP):",
            expdperr[expdperr < 0].abs().sum()
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


def match_true(df_true, dfs_to_match):
    for c in (c for c in df_true.columns if "true" in c):
        for df in dfs_to_match:
            df.loc[:, c] = df_true.loc[:, c]
