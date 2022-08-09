source("R/utils.R")

regression_tables <- function(experiment_name, sampling_only, trials) {
  #' Make tables for various regression specifications
  #' for a given experiment for a given number of trials.
  #' If `sampling_only`, run regressions for just data deviations.
  if (missing(trials)) {
    trials <- 100
  }

  # load experiment
  experiment <- load_experiment(experiment_name) %>% filter(trial < trials)

  # clean for regression
  df_reg <- clean_for_reg(experiment, sampling_only)

  print("- OLS")
  lm_formula <- lm(
    misalloc ~
      log(true_pop_total) +
      true_children_total +
      true_children_poverty,
    data = df_reg
  )
  lm_final <- lm(
    misalloc ~
      log(pop_density) +
      hhi +
      prop_white +
      prop_hispanic +
      median_income_est +
      renter_occupied_housing_tenure_pct,
    data = df_reg
  )
  lm_extra <- lm(
    misalloc ~
      log(pop_density) +
      hhi +
      prop_white +
      prop_hispanic +
      median_income_est +
      renter_occupied_housing_tenure_pct +
      log(true_pop_total) +
      true_children_total +
      true_children_poverty +
      not_a_u_s_citizen_u_s_citizenship_status_pct +
      average_household_size_of_renter_occupied_unit_housing_tenure_est,
    data = df_reg
  )
  # OLS table
  stargazer(
    lm_formula, lm_extra, lm_final,
    type = "latex",
    column.labels = c(
      "Formula components",
      "All variables",
      "Demographic variables"
    ),
    dep.var.labels = "Misallocation (deviated minus true)",
    covariate.labels = c(
      "Log population density",
      "Racial homogeneity (HHI)",
      "Proportion White",
      "Proportion Hispanic",
      "Median income",
      "% households renting",
      "Log total population",
      "Total # children",
      "Total # children in poverty",
      "% not a U.S. citizen",
      "Avg. renter's household size"
    ),
    out = sprintf("plots/tables/%s_ols.tex", experiment_name)
  )

  # Get GAM results
  print("- GAM")
  gam_mr <- get_gam("baseline", sampling_only, TRUE, experiment)
  gam_table(gam_mr, sprintf("plots/tables/%s_gam.tex", experiment_name))
  # Compare GAM and OLS
  print(
    xtable(anova(lm_final, gam_mr)),
    file=sprintf("plots/tables/%s_anova_lm_gam.tex", experiment_name)
  )

  print("- GAM with interaction")
  gam_mr_interact <- gam(
    misalloc ~
      te(
        prop_white,
        median_income_est
      ) +
      s(log(pop_density), bs="tp") +
      s(hhi, bs = "tp") +
      s(prop_hispanic, bs = "tp") +
      s(renter_occupied_housing_tenure_pct, bs = "tp"),
    data = df_reg
  )
  sink(sprintf("plots/tables/%s_interact.tex", experiment_name))
  gamtabs_summary(gam_mr_interact, label = "Demographic GAN with interactions")
  sink()
  # Compare GAM with and without interaction terms
  stargazer(
    anova.gam(gam_mr, gam_mr_interact, test = "F"),
    type = "latex",
    summary = FALSE,
    out = sprintf("plots/tables/%s_anova_gam_interact.tex", experiment_name)
  )
}

regression_tables("baseline", FALSE, 10)