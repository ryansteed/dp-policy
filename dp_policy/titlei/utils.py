import pandas as pd
from math import floor, ceil


def get_saipe(path):
    saipe = pd.read_excel(path, header=2).set_index(["State FIPS Code", "District ID"])
    saipe["median_cv"] = saipe.apply(lambda x: median_cv(x["Estimated Total Population"]), axis=1)
    return saipe


def get_race(path):
    """Method for loading and formatting ACS data by school district from NCES website.
    https://nces.ed.gov/programs/edge/TableViewer/acs/2018

    Args:
        path (str): path to txt file downloaded form the link above
    """
    race = pd.read_csv(path, sep="|")
    # strip out NA district ID's
    race = race[race["LEAID"] != 'N']
    # separate LEAID into FIPS code and district ID
    race["District ID"] = race["LEAID"].str[2:].astype(int)
    race["State FIPS Code"] = race["LEAID"].str[:2].astype(int)
    race = race.set_index(["State FIPS Code", "District ID"])
    race = race.drop(columns=["GeoId", "Geography", "Year", "Iteration", "LEAID"])
    race = race.rename(columns = {
        col: f"race_{col[7:]}" for col in race.columns
    })
    return race


def get_sppe(path):
    states = { 'Alabama': 'AL', 'Alaska': 'AK', 'American Samoa': 'AS', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Guam': 'GU', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Northern Mariana Islands':'MP', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virgin Islands': 'VI', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY' }
    states = pd.DataFrame(states.items(), columns=["state", "abbrv"])
    
    # quirk of original data file - need to change DC's name for join
    states[states.state == "District of Columbia"] = "District Of Columbia Public Schools"

    sppe = pd.read_excel(path, header=2).rename(columns={"Unnamed: 0": "state"})[["state", "ppe"]]

    return sppe.merge(states, on="state", how="right")


def median_cv(total_pop):
    # based on the table given here https://www.census.gov/programs-surveys/saipe/guidance/district-estimates.html
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


def weighting(eligible, pop):
    """
    Gradated weighting algorithm given in [Sonnenberg](https://nces.ed.gov/surveys/annualreports/pdf/titlei20160111.pdf).
    
    Returns weighted eligibility counts.
    """
    
    # calculate weighted count based on counts
    wec_counts = 0
    for r, w in {(1, 691): 1.0, (692, 2262): 1.5, (2263,7851): 2.0, (7852, 35514): 2.5, (35514, None): 3.0}.items():
        if r[1] is not None and eligible > r[1]: 
            wec_counts += (r[1] - r[0] + 1) * w
        elif eligible >= r[0]: 
            wec_counts += (eligible - r[0] + 1) * w
    
    # calculate weighted count based on proportions
    wec_props = 0
    for r, w in {(0, 0.1558): 1.0, (0.1558, 0.2211): 1.75, (0.2211,0.3016): 2.5, (0.3016, 0.3824): 3.25, (0.3824, None): 4.0}.items():
        upper = floor(r[1]*pop) if r[1] is not None else None
        lower = ceil(r[0]*pop)
        
        if upper is not None and eligible > upper: 
            wec_props += (upper - lower) * w
        elif eligible >= lower: 
            wec_props += (eligible - lower) * w
    
    # take the higher weighted eligibility count
    return max(wec_counts, wec_props)
