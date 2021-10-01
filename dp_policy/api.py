import pandas as pd


def titlei_data(
    saipe, mechanism, sppe, *mech_args,
    verbose=True, **mech_kwargs
):
    grants = saipe.rename(columns={
        "Estimated Total Population": "true_pop_total",
        "Estimated Population 5-17": "true_children_total",
        "Estimated number of relevant children 5 to 17 years old in poverty"
        " who are related to the householder": "true_children_poverty"
    })
    pop_total, children_total, children_poverty = mechanism.poverty_estimates(
        *mech_args, **mech_kwargs
    )
    grants["est_pop_total"] = pop_total
    grants["est_children_total"] = children_total
    grants["est_children_poverty"] = children_poverty

    # BIG ASSUMPTION, TODO: revisit later
    grants["true_children_eligible"] = grants.true_children_poverty
    grants["est_children_eligible"] = grants.est_children_poverty

    # join in SPPE
    grants = grants.reset_index()\
        .merge(sppe, left_on="State Postal Code", right_on="abbrv")\
        .drop(columns=['abbrv', 'state']).rename(columns={'ppe': 'sppe'})\
        .set_index(["State FIPS Code", "District ID"])

    if verbose:
        print(
            "[WARN] Dropping districts with missing SPPE data:",
            grants[grants.sppe.isna()]['Name'].values
        )
    grants = grants.dropna(subset=["sppe"])
    grants.sppe = grants.sppe.astype(float)

    return grants


def titlei_funding(
    allocator, saipe, mechanism, sppe,
    uncertainty=True, allocator_kwargs={},
    **grants_kwargs
):
    """
    Returns augmented SAIPE dataframe with randomized estimates and
    true/randomized grant amounts.
    """
    alloc = allocator(
        titlei_data(saipe, mechanism, sppe, **grants_kwargs),
        **allocator_kwargs
    )
    return alloc.allocations(uncertainty=uncertainty)
