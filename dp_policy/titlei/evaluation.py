from dp_policy.titlei.utils import get_acs_unified

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
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
        "Foreign born (PLACE OF BIRTH) - est",
        "Foreign born (PLACE OF BIRTH) - pct",
        "Not a U.S. citizen (U.S. CITIZENSHIP STATUS) - est",
        "Not a U.S. citizen (U.S. CITIZENSHIP STATUS) - pct"
    ]
    # add language isolation
    variables += [
        'Language other than English (LANGUAGE SPOKEN AT HOME) - est',
        'Language other than English (LANGUAGE SPOKEN AT HOME) - pct'
    ]
    # add renters vs. homeowners (housing security)
    variables += [
        'Renter-occupied (HOUSING TENURE) - est',
        'Renter-occupied (HOUSING TENURE) - pct',
        'Average household size of renter-occupied unit (HOUSING TENURE) - est'
    ]
    if verbose:
        print(variables)

    # adding geographic area
    geo = get_geography()

    grants = results.join(acs[variables], how="inner")
    if verbose:
        print(grants.shape)
    grants = grants.join(geo["ALAND"])
    if verbose:
        print(grants.shape)
        print(
            grants[
                grants["Total population (RACE) - est"].isna()
            ].groupby(["State FIPS Code", "District ID"]).groups.keys()
        )
    if save_path:
        grants.to_csv(save_path)
    return grants


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
        (results.est_eligible_targeted & ~results.true_eligible_targeted) \
        | (results.est_eligible_basic & ~results.true_eligible_basic) \
        | (
            results.est_eligible_concentration
            & ~results.true_eligible_concentration
        )
    results["became_ineligible"] = \
        (~results.est_eligible_targeted & results.true_eligible_targeted) \
        | (~results.est_eligible_basic & results.true_eligible_basic) \
        | (
            ~results.est_eligible_concentration
            & results.true_eligible_concentration
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
        (results.dpest_eligible_targeted & ~results.true_eligible_targeted)\
        | (results.dpest_eligible_basic & ~results.true_eligible_basic)\
        | (
            results.dpest_eligible_concentration
            & ~results.true_eligible_concentration
        )
    results["became_ineligible_dp"] = \
        (~results.dpest_eligible_targeted & results.true_eligible_targeted)\
        | (~results.dpest_eligible_basic & results.true_eligible_basic)\
        | (
            ~results.dpest_eligible_concentration
            & results.true_eligible_concentration
        )
    results["dp_marginal"] = \
        results["error_dp_per_child"] - results["error_per_child"]

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
        ]].groupby(["State FIPS Code", "District ID"]).mean(),
        how="inner"
    )
    joined.loc[
        np.isinf(joined["error_per_child"]), "error_per_child"
    ] = np.nan
    joined.loc[
        np.isinf(joined["error_per_child_eligible"]),
        "error_per_child_eligible"
    ] = np.nan
    joined.loc[
        np.isinf(joined["error_dp_per_child"]), "error_dp_per_child"
    ] = np.nan
    joined.loc[
        np.isinf(joined["error_dp_per_child_eligible"]),
        "error_dp_per_child_eligible"
    ] = np.nan

    return joined


def plot_treatments(
    treatments, x_func, plot_method, plot_kwargs,
    filename=None,
    xlab=None,
    ylab="Smoothed density",
    grant="total",
    epsilon=0.1,
    delta=0.0,
    mean_line=False
):
    palette = sns.color_palette()
    for i, (treatment, df_raw) in enumerate(treatments.items()):
        df = df_raw.loc[(
                slice(None), delta, epsilon, slice(None), slice(None)
            ), :].copy()
        df.loc[:, "misalloc"] = \
            df[f"dpest_grant_{grant}"] - df[f"true_grant_{grant}"]
        df.loc[:, "misalloc_sq"] = np.power(df["misalloc"], 2)
        if grant == "total":
            df["lost_eligibility"] = \
                (
                    df["dpest_eligible_basic"] &
                    ~df["true_eligible_basic"]
                ) |\
                (
                    df["dpest_eligible_concentration"] &
                    ~df["true_eligible_concentration"]
                ) |\
                (
                    df["dpest_eligible_targeted"] &
                    ~df["true_eligible_targeted"]
                )
        else:
            df["lost_eligibility"] = \
                ~df["dpest_eligible_{}".format(grant)] \
                & df["true_eligible_{}".format(grant)]
        plot_kwargs.update({
            'label': treatment,
            'color': palette[i]
        })
        x = x_func(df)
        plot_method(x, **plot_kwargs)
        if mean_line:
            plt.axvline(x.mean(), color=palette[i], linestyle='dashed')

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='upper right')
    if filename:
        plt.savefig(f"../plots/bootstrap/{filename}.png", dpi=100)
    plt.show()
    plt.close()


def cube(x):
    return np.sign(x)*np.power(np.abs(x), 1/3)


def heatmap(
    data, label=None, title=None, transform='cube', theme="RdBu",
    y="error_dp_per_child", vcenter=0, file=None,
    figsize=(10, 5), bar_location='bottom', min=None, max=None
):
    fig, ax = plt.subplots(1, figsize=figsize, dpi=300)

    if transform == 'cube':
        data.loc[:, y] = cube(data[y])
    if transform == 'log':
        data.loc[:, y] = np.where(data[y] == 0, 0, np.log(data[y]))

    # Create colorbar as a legend
    if min is None and max is None:
        min = data[y].min()
        max = data[y].max()

    if vcenter is not None and transform != 'log':
        norm = pltc.TwoSlopeNorm(vcenter=0, vmin=min, vmax=max)
    else:
        norm = pltc.Normalize(vmin=min, vmax=max)
    sm = plt.cm.ScalarMappable(cmap=theme, norm=norm)

    data.plot(
        column=y, cmap=theme, norm=norm, ax=ax, linewidth=0.05, edgecolor='0.2'
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
        plt.savefig(f"../plots/geo/{file}", dpi=300)
        plt.close()
    else:
        plt.show()


def save_treatments(treatments, experiment_name):
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
  epsilon=0.1, delta=0.0, mapvar="error_per_child"
):
    print("Comparing at eps=", epsilon)
    for treatment, df in treatments.items():
        print("#", treatment)
        print("True budget:", df[f"true_grant_total"].sum())
        print("DP est budget:",  df[f"dpest_grant_total"].sum())
        print(
            "RMSE:",
            np.sqrt(((df.dpest_grant_total - df.true_grant_total) ** 2).mean())
        )
        df["became_ineligible"] = \
            (~df.dpest_eligible_targeted & df.true_eligible_targeted)\
            | (~df.dpest_eligible_basic & df.true_eligible_basic)\
            | (
                ~df.dpest_eligible_concentration
                & df.true_eligible_concentration
            )
        print(
            "Avg prop. districts erroneously ineligible:",
            df.became_ineligible.mean()
        )

    # compare RMSE - mean and hist
    plot_treatments(
        treatments,
        lambda df: np.sqrt(df.groupby('trial')["misalloc_sq"].mean()),
        sns.kdeplot,
        dict(bw_method=0.5, fill=True),
        # filename=f"rmse_{grant}",
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
                df.loc[:, delta, epsilon, :, :]
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
            y="error_per_child",
            label="Misallocation per child (cube root)",
            title=treatment,
            # file="misalloc_nation_sampling.png",
            figsize=(15, 10),
            bar_location='right',
            min=ymin,
            max=ymax
        )


def match_true(df_true, dfs_to_match):
    for c in (c for c in df_true.columns if "true" in c):
        for df in dfs_to_match:
            df.loc[:, c] = df_true.loc[:, c]
