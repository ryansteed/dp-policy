import pandas as pd
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


def average_saipe(paths):
    print(paths)
    saipes = [get_saipe(path) for path in paths]
    combined = pd.concat(saipes)
    combined = combined \
        .groupby(["State FIPS Code", "District ID"]) \
        .agg({
            'State Postal Code': 'first',
            'Name': 'first',
            'Estimated Total Population': 'mean',
            'Estimated Population 5-17': 'mean',
            'Estimated number of relevant children 5 to 17 years old '
            'in poverty who are related to the householder': 'mean',
            'cv': 'mean'
        })
    combined["cv"] = combined.apply(
        lambda x: median_cv(x["Estimated Total Population"]),
        axis=1
    )
    return combined


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
