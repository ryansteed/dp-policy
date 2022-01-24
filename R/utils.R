library(stargazer)
library(tidyverse)
library(ggpubr)
library(janitor)
library(mgcv)
library(lmtest)
library(mgcViz)
library(tidymv)
library(MASS)
library(cowplot)
library(grid)
library(gridExtra)
library(data.table)

clean = function(df) {
  df_clean = df %>%
    clean_names() %>%
    mutate_at(
      vars(matches("race_pct$")), list(`children`= ~(. / 100 * true_children_total))
    ) %>% mutate(
      prop_white = white_race_pct / 100,
      nonwhite_children = true_children_total * prop_white,
      not_a_u_s_citizen_u_s_citizenship_status_pct = as.numeric(not_a_u_s_citizen_u_s_citizenship_status_pct),
      average_household_size_of_renter_occupied_unit_housing_tenure_est = as.numeric(gsub(',', '', average_household_size_of_renter_occupied_unit_housing_tenure_est)),
      median_income_est = as.numeric(gsub('[\\+]|[,]', '', as.character(median_household_income_dollars_income_and_benefits_in_2019_inflation_adjusted_dollars_est))),
      pop_density = true_pop_total / aland
    ) %>% mutate(
      # misalloc in terms of true grant total - TODO make sure this matches, normalize before this
      misalloc_dp = dp_grant_total - true_grant_total,
      misalloc_sampling = est_grant_total - true_grant_total,
      misalloc_dp_sampling = dpest_grant_total - true_grant_total
    )
  return(df_clean)
}

summarise_trials = function(df) {
  grouped = df %>% 
    group_by(treatment, state_fips_code, district_id) %>%
    summarise_at(
      c("misalloc_dp", "misalloc_sampling", "misalloc_dp_sampling"),
      mean,
      na.rm=TRUE
    ) %>%
    ungroup() %>%
    left_join(
      df %>% dplyr::select(
        state_fips_code, 
        district_id, 
        nonwhite_children,
        true_grant_total,
        true_pop_total,
        true_children_eligible,
        true_children_total, 
        prop_white,
        ends_with("race_pct"),
        language_other_than_english_language_spoken_at_home_pct,
        language_other_than_english_language_spoken_at_home_est,
        foreign_born_place_of_birth_est,
        foreign_born_place_of_birth_pct,
        not_a_u_s_citizen_u_s_citizenship_status_est,
        not_a_u_s_citizen_u_s_citizenship_status_pct,
        renter_occupied_housing_tenure_pct,
        renter_occupied_housing_tenure_est,
        average_household_size_of_renter_occupied_unit_housing_tenure_est,
        median_income_est,
        pop_density
      ) %>% unique(),
      by=c("state_fips_code", "district_id")
    ) %>%
    mutate(
      misalloc_dp_per_child = misalloc_dp / true_children_total,
      misalloc_sampling_per_child = misalloc_sampling / true_children_total,
      misalloc_dp_sampling_per_child = misalloc_dp_sampling / true_children_total
    )
  # optional - filter to only look at edge cases
  # grouped = grouped  %>%
  #   filter(
  #     misalloc > mean(misalloc_per_child) + 2*sd(misalloc_per_child) | misalloc_per_child < mean(misalloc_per_child) - 2*sd(misalloc_per_child)
  #   )
  return(grouped) 
}

cuberoot = function(x) {
  return(sign(x)*abs(x)^(1/3))
}

race_comparison = function(comparison, kind) {
  if (kind == "hispanic") {
    comparison = comparison %>%
      dplyr::select(ends_with("hispanic_or_latino_and_race_pct") | !ends_with("race_pct"))
  } else if (kind == "race_aggregate") {
    comparison = comparison %>%
      mutate(
        white_agg_race_pct = white_race_pct,
        black_or_african_american_agg_race_pct = black_or_african_american_race_pct,
        tribal_grouping_agg_race_pct = rowSums(across(ends_with("tribal_grouping_race_pct"))),
        asian_agg_race_pct = asian_race_pct,
        pacific_islander_agg_race_pct = rowSums(across(c(
          "native_hawaiian_race_pct",
          "guamanian_or_chamorro_race_pct",
          "samoan_race_pct"
        ))),
        some_other_race_agg_race_pct = some_other_race_race_pct,
        two_or_more_races_agg_race_pct = two_or_more_races_race_pct
      ) %>%
      dplyr::select(ends_with("agg_race_pct") | !ends_with("race_pct"))
  } else {
    comparison = comparison %>%
      dplyr::select(!ends_with("hispanic_or_latino_and_race_pct")) %>%
      dplyr::select(
        !asian_race_pct
      )
  }
  
  remainder = comparison %>%
    dplyr::select(ends_with("race_pct")) %>%
    mutate(sum = rowSums(across(everything())))
  
  # per the census, should add to 100 - https://www.census.gov/quickfacts/fact/note/US/RHI625219
  print(remainder %>% filter(sum > 105.0))
  
  # total misalloc per student ("burden")
  comparison = comparison %>%
    # mutate(other_race_pct = 100-rowSums(across(ends_with("race_pct")))) %>%
    # mutate(other_race_pct = if_else(other_race_pct < 0, 0, other_race_pct)) %>%
    pivot_longer(
      ends_with("race_pct"),
      values_to="race_pct",
      names_to="race"
    ) %>%
    mutate(
      race=recode_factor(
        str_replace(race, "_race_pct", ""),
        `sioux_tribal_grouping` = "Sioux tribal grouping",
        `chippewa_tribal_grouping` = "Chippewa tribal grouping",
        `navajo_tribal_grouping` = "Navajo tribal grouping",
        `cherokee_tribal_grouping` = "Cherokee tribal grouping",
        `guamanian_or_chamorro` = "Guamanian or Chamorro",
        `native_hawaiian` = "Native Hawaiian",
        `white` = "White",
        `samoan` = "Samoan",
        `two_or_more_races` = "Two or more races",
        `filipino` = "Filipino",
        `japanese` = "Japanese",
        `black_or_african_american` = "Black or African American",
        `other_asian` = "Other Asian",
        `some_other_race` = "Some other race",
        `korean` = "Korean",
        `vietnamese` = "Vietnamese",
        `asian_indian` = "Asian Indian",
        `chinese` = "Chinese",
        `tribal_grouping_agg` = "Tribal grouping",
        `pacific_islander_agg` = "Pacific islander",
        `white_agg` = "White",
        `two_or_more_races_agg` = "Two or more races",
        `black_or_african_american_agg` = "Black or African American",
        `some_other_race_agg` = "Some other race",
        `asian_agg` = "Asian",
        `not_hispanic_or_latino_hispanic_or_latino_and` = "Not Hispanic or Latino",
        `mexican_hispanic_or_latino_and` = "Mexican",
        `puerto_rican_hispanic_or_latino_and` = "Puerto Rican",
        `other_hispanic_or_latino_hispanic_or_latino_and` = "Other Hispanic or Latino",
        `cuban_hispanic_or_latino_and` = "Cuban"
      )
    ) %>%
    # assuming misallocation is spread evenly across children,
    mutate(
      benefit_dp = misalloc_dp_per_child / race_pct * 100,
      benefit_sampling = misalloc_sampling_per_child / race_pct * 100,
      benefit_dp_sampling = misalloc_dp_sampling_per_child / race_pct * 100
    )
  
  comparison_all = comparison %>%
    group_by(treatment, race) %>%
    mutate(
      children_of_race = round(true_children_total * race_pct / 100),
      children_of_race_eligible = round(true_children_eligible * race_pct / 100),
      dp_benefit_per_child = sum(misalloc_dp * race_pct / 100) / sum(children_of_race),
      dp_benefit_per_child_eligible = sum(misalloc_dp * race_pct / 100) / sum(children_of_race_eligible),
      sampling_benefit_per_child = sum(misalloc_sampling * race_pct / 100) / sum(children_of_race),
      sampling_benefit_per_child_eligible = sum(misalloc_sampling * race_pct / 100) / sum(children_of_race_eligible),
      dp_sampling_benefit_per_child = sum(misalloc_dp_sampling * race_pct / 100) / sum(children_of_race),
      dp_sampling_benefit_per_child_eligible = sum(misalloc_dp_sampling * race_pct / 100) / sum(children_of_race_eligible)
    )  %>%
    ungroup()
  
  # comparison_mean = comparison_all %>%
  #   group_by(treatment, race) %>%
  #   summarise(
  #     dp_benefit_per_child = sum(misalloc_dp * race_pct / 100) / sum(children_of_race),
  #     dp_benefit_per_child_eligible = sum(misalloc_dp * race_pct / 100) / sum(children_of_race_eligible),
  #     sampling_benefit_per_child = sum(misalloc_sampling * race_pct / 100) / sum(children_of_race),
  #     sampling_benefit_per_child_eligible = sum(misalloc_sampling * race_pct / 100) / sum(children_of_race_eligible),
  #     dp_sampling_benefit_per_child = sum(misalloc_dp_sampling * race_pct / 100) / sum(children_of_race),
  #     dp_sampling_benefit_per_child_eligible = sum(misalloc_dp_sampling * race_pct / 100) / sum(children_of_race_eligible)
  #   )
  
  sorted = comparison_all %>%
    mutate(race = fct_reorder(race, sampling_benefit_per_child)) %>%
    distinct(treatment, race, sampling_benefit_per_child, dp_sampling_benefit_per_child)
  
  return(sorted)
}