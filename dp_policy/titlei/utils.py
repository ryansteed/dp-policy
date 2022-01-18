import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
import pickle
import re
import os
from math import floor, ceil


def get_saipe(path):
    saipe = pd.read_excel(path, header=2)\
        .set_index(["State FIPS Code", "District ID"])
    saipe["cv"] = saipe.apply(
        lambda x: median_cv(x["Estimated Total Population"]),
        axis=1
    )
    return saipe


def get_acs_data(path, name):
    """Method for loading and formatting ACS data by school district from NCES
    [website](https://nces.ed.gov/programs/edge/TableViewer/acs/2018).

    Args:
        path (str): path to txt file downloaded form the link above
    """
    data = pd.read_csv(path, sep="|")
    # strip out NA district ID's
    # data = data[data["LEAID"] != 'N']
    # separate LEAID into FIPS code and district ID
    data["District ID"], data["State FIPS Code"] = \
        split_leaids(data.LEAID)
    data = data.set_index(["State FIPS Code", "District ID"])
    data = data.drop(
        columns=["GeoId", "Geography", "Year", "Iteration"]
    )

    varnames = pd.read_excel(
        "../data/discrimination/ACS-ED_2015-2019_RecordLayouts.xlsx",
        sheet_name="CDP_ChildPop",
        index_col=0
    )
    drop = []
    new = {}
    for c in data.columns:
        try:
            new[c] = "{} ({}) - {}".format(
                varnames.loc[c].vlabel.split(";")[-1].lstrip(),
                varnames.loc[c].vlabel.split(";")[2].lstrip(),
                re.sub(r'\d+', '', c.split("_")[-1])
            )
        except KeyError:
            drop.append(c)
    data = data.drop(columns=drop).rename(columns=new)
    return data


def split_leaids(leaids: pd.Series):
    # get the last seven digits of the ID
    leaids = leaids.astype(str).str[-7:]
    return leaids.str[-5:].astype(int), leaids.str[:-5].astype(int)


def get_sppe(path):
    states = {
        'Alabama': 'AL', 'Alaska': 'AK', 'American Samoa': 'AS',
        'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA',
        'Guam': 'GU', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
        'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY',
        'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
        'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND',
        'Northern Mariana Islands': 'MP', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR',
        'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virgin Islands': 'VI', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    states = pd.DataFrame(states.items(), columns=["state", "abbrv"])

    # quirk of original data file - need to change DC's name for join
    states[states.state == "District of Columbia"] = \
        "District Of Columbia Public Schools"

    sppe = pd.read_excel(path, header=2)\
        .rename(columns={"Unnamed: 0": "state"})[["state", "ppe"]]

    return sppe.merge(states, on="state", how="right")


def median_cv(total_pop):
    # based on the table given here
    # https://www.census.gov/programs-surveys/saipe/guidance/district-estimates.html
    if total_pop <= 2500:
        return 0.67
    elif total_pop <= 5000:
        return 0.42
    elif total_pop <= 10000:
        return 0.35
    elif total_pop <= 20000:
        return 0.28
    elif total_pop <= 65000:
        return 0.23
    return 0.15


def get_allocation_data(dir: str, header=1):
    """Fetch true Title I allocations.

    Args:
        dir (str): directory containing state-level files.
    """
    data = []
    for f in os.listdir(dir):
        state = pd.read_excel(
            os.path.join(dir, f),
            header=header,
            names=["LEAID", "District", "HistAlloc"],
            usecols=[0, 1, 2],
            skipfooter=7,
            na_values=["No Data", "End of Table", ""]
        )
        state["state"] = f
        data.append(state)
    df = pd.concat(data)
    df["District ID"], df["State FIPS Code"] = \
        split_leaids(df.LEAID)
    return df.set_index(["State FIPS Code", "District ID"])


def weighting(eligible, pop):
    """
    Gradated weighting algorithm given in
    [Sonnenberg](https://nces.ed.gov/surveys/annualreports/pdf/titlei20160111.pdf).

    Returns weighted eligibility counts.
    """

    # calculate weighted count based on counts
    wec_counts = 0
    for r, w in {
        (1, 691): 1.0,
        (692, 2262): 1.5,
        (2263, 7851): 2.0,
        (7852, 35514): 2.5,
        (35514, None): 3.0
    }.items():
        if r[1] is not None and eligible > r[1]:
            wec_counts += (r[1] - r[0] + 1) * w
        elif eligible >= r[0]:
            wec_counts += (eligible - r[0] + 1) * w

    # calculate weighted count based on proportions
    wec_props = 0
    for r, w in {
        (0, 0.1558): 1.0,
        (0.1558, 0.2211): 1.75,
        (0.2211, 0.3016): 2.5,
        (0.3016, 0.3824): 3.25,
        (0.3824, None): 4.0
    }.items():
        upper = floor(r[1]*pop) if r[1] is not None else None
        lower = ceil(r[0]*pop)

        if upper is not None and eligible > upper:
            wec_props += (upper - lower) * w
        elif eligible >= lower:
            wec_props += (eligible - lower) * w

    # take the higher weighted eligibility count
    return max(wec_counts, wec_props)


def get_acs_unified(verbose=False):
    demographics = get_acs_data(
        "../data/discrimination/CDP05.txt",
        "demo"
    )
    social = get_acs_data(
        "../data/discrimination/CDP02.txt",
        "social"
    )
    economic = get_acs_data(
        "../data/discrimination/CDP03.txt",
        "social"
    )
    housing = get_acs_data(
        "../data/discrimination/CDP04.txt",
        "housing"
    )
    if verbose:
        print(demographics.shape)
        print(social.shape)
        print(economic.shape)
        print(housing.shape)
    return demographics\
        .join(social, lsuffix="_demo", rsuffix="_social", how="inner")\
        .join(economic, rsuffix="_econ", how="inner")\
        .join(housing, rsuffix="_housing", how="inner")


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


def compare_treatments(treatments, epsilon=0.1, delta=0.0):
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

    # compare likelihood of ineligible
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
