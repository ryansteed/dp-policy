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
      mean
    ) %>%
    ungroup()  %>%
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
  print(sprintf("%d rows with overfull race percent", remainder %>% filter(sum > 105.0) %>% nrow()))
  
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
      benefit_dp = ifelse(race_pct > 0.0, misalloc_dp_per_child / race_pct * 100, NA),
      benefit_sampling = ifelse(race_pct > 0.0, misalloc_sampling_per_child / race_pct * 100, NA),
      benefit_dp_sampling = ifelse(race_pct > 0.0, misalloc_dp_sampling_per_child / race_pct * 100, NA)
    )
  
  comparison_all = comparison %>%
    group_by(treatment, race) %>%
    mutate(
      children_of_race = round(true_children_total * race_pct / 100),
      children_of_race_eligible = round(true_children_eligible * race_pct / 100),
      dp_benefit_per_child = sum(misalloc_dp * race_pct / 100, na.rm=TRUE) / sum(children_of_race, na.rm=TRUE),
      dp_benefit_per_child_eligible = sum(misalloc_dp * race_pct / 100, na.rm=TRUE) / sum(children_of_race_eligible, na.rm=TRUE),
      sampling_benefit_per_child = sum(misalloc_sampling * race_pct / 100, na.rm=TRUE) / sum(children_of_race, na.rm=TRUE),
      sampling_benefit_per_child_eligible = sum(misalloc_sampling * race_pct / 100, na.rm=TRUE) / sum(children_of_race_eligible, na.rm=TRUE),
      dp_sampling_benefit_per_child = sum(misalloc_dp_sampling * race_pct / 100, na.rm=TRUE) / sum(children_of_race, na.rm=TRUE),
      dp_sampling_benefit_per_child_eligible = sum(misalloc_dp_sampling * race_pct / 100, na.rm=TRUE) / sum(children_of_race_eligible, na.rm=TRUE)
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

load_experiment = function(name) {
  raw = fread(sprintf("results/policy_experiments/%s_discrimination_laplace_eps=0.1.csv", name))
  df = clean(raw)
  return(df)
}

plot_race = function(experiment, name) {
  grouped = summarise_trials(experiment)
  comparison = race_comparison(grouped, "race_aggregate")
  
  plt = ggplot(comparison, aes(x=race, y=sampling_benefit_per_child)) +
    geom_col(position="dodge", aes(fill=treatment), color="black") +
    geom_errorbar(
      aes(
        # x=race,
        # y=dp_sampling_benefit_per_child,
        ymin=sampling_benefit_per_child, 
        ymax=dp_sampling_benefit_per_child,
        # linetype="",
        color="",
        fill=treatment
      ),
      position=position_dodge(width=0.9),
      size=0.5,
      width=0.5
    ) +
    ylab("Race-weighted misallocation per eligible child") +
    scale_colour_manual(values=c("black")) +
    coord_flip() +
    xlab("Census Race Category") +
    guides(
      fill = guide_legend(ncol=3)
    ) +
    labs(
      fill = "Data error",
      color = "+ DP error"
    ) +
    theme(
      legend.position = "top",
      legend.box="vertical"
    )
  print(plt)
  ggsave(sprintf("plots/race/misalloc_%s.png", name), dpi=300, width=5, height=6)
}

sq_share = function(x) {
  return((x/100)^2)
}

clean_for_reg = function(df) {
  df = df %>%
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
      two_or_more_races_agg_race_pct = two_or_more_races_race_pct,
      hhi = sq_share(white_race_pct) +
        sq_share(black_or_african_american_race_pct) +
        sq_share(cherokee_tribal_grouping_race_pct) +
        sq_share(chippewa_tribal_grouping_race_pct) +
        sq_share(navajo_tribal_grouping_race_pct) +
        sq_share(sioux_tribal_grouping_race_pct) +
        sq_share(asian_indian_race_pct) +
        sq_share(chinese_race_pct) +
        sq_share(filipino_race_pct) +
        sq_share(japanese_race_pct) +
        sq_share(korean_race_pct) +
        sq_share(vietnamese_race_pct) +
        sq_share(other_asian_race_pct) +
        sq_share(native_hawaiian_race_pct) +
        sq_share(guamanian_or_chamorro_race_pct) +
        sq_share(samoan_race_pct) +
        sq_share(some_other_race_race_pct) +
        sq_share(two_or_more_races_race_pct),
      prop_hispanic = 1-not_hispanic_or_latino_hispanic_or_latino_and_race_pct/100
    ) %>%
    dplyr::select(
      -ends_with("hispanic_or_latino_and_race_pct"),
      -asian_race_pct
    )
  # filter(trial < 10)
  return(df)
}

run_regs = function(df, sampling_only, savepath) {
  df_reg = clean_for_reg(df)
  
  if (sampling_only) {
    df_abbrv = df_reg %>% mutate(misalloc = misalloc_sampling)
  } else {
    df_abbrv = df_reg %>% mutate(misalloc = misalloc_dp_sampling)
  }
  
  lm_mr = lm(
    misalloc ~ log(pop_density) +
      median_income_est +
      hhi +
      prop_white +
      prop_hispanic +
      renter_occupied_housing_tenure_pct,
    # true_children_total + 
    # true_children_poverty +
    # not_a_u_s_citizen_u_s_citizenship_status_pct +
    # average_household_size_of_renter_occupied_unit_housing_tenure_est,
    data=df_abbrv
  )
  print(summary(lm_mr))
  
  gam_mr = gam(
    misalloc ~
      # s(true_children_total, bs="tp") +
      # s(true_children_poverty, bs="tp") +
      s(log(pop_density), bs="tp") +
      s(hhi, bs="tp") +
      s(prop_white, bs="tp") +
      s(prop_hispanic, bs="tp") +
      s(median_income_est, bs="tp") +
      # s(not_a_u_s_citizen_u_s_citizenship_status_pct, bs="tp") +
      s(renter_occupied_housing_tenure_pct, bs="tp"),
    # s(average_household_size_of_renter_occupied_unit_housing_tenure_est, bs="tp"),
    method="REML", # restricted MLE
    data=df_abbrv
  )
  print(summary(gam_mr))
  print(anova(lm_mr, gam_mr))
  
  if (!missing(savepath)) {
    saveRDS(gam_mr, savepath)
  }
  
  return(gam_mr)
}

get_gam = function(name, sampling_only, from_cache, df) {
  savepath = sprintf("results/regressions/%s_sampling=%s.rds", name, toString(sampling_only))
  # print(savepath)
  if (file.exists(savepath) & from_cache) {
    return(readRDS(savepath))
  }
  else if (!missing(df)) {
    return(run_regs(df, sampling_only, savepath))
  }
  else {
    print("Missing df, could not run regressions.")
  }
}

get_gam_viz = function(plotname, from_cache, gam) {
  savepath = sprintf("results/regressions/viz_%s.rds", plotname)
  if (file.exists(savepath) & from_cache) {
    return(readRDS(savepath))
  }
  else if (!missing(gam)) {
    viz = getViz(gam)
    saveRDS(viz, savepath)
    return(viz)
  }
  else {
    print("Missing gam, could not get viz.")
  }
}

plot_gam = function(viz, plotname) {
  labels = c(
    # "# children",
    # "# children in poverty",
    "Log population density",
    "Racial homogeneity (HHI)",
    "% white-only",
    "% hispanic",
    "Median income",
    "% renter-occupied housing"
    # "Average size of renter household"
  )
  lower_limits = c(
    -1500000, # pop density
    -70000, # HHI
    -120000, # white-only
    -30000, # hispanic
    -50000, # income
    -200000 # housing
  )
  upper_limits = c(
    300000, # pop density
    30000, # HHI
    70000, # white-only
    80000, # hispanic
    100000, # income
    400000 # housing
  )
  
  level = 0.95
  mul = qnorm((level+1)/2)
  plt = function(i) {
    plot_obj = plot(sm(viz, i))
    upper = max(plot_obj$data$fit$y + mul*plot_obj$data$fit$se)
    lower = min(plot_obj$data$fit$y - mul*plot_obj$data$fit$se)
    p = plot_obj +
      theme_minimal() +
      # ylab("Smoothed effect (in terms of $$ misallocated)") +
      l_points(shape = 19, size = 0.5, alpha = 0.05, color="blue") +
      # l_dens("joint") +
      l_ciPoly(level=level, alpha=0.5, size=0.25) +
      l_ciLine(level=level) +
      l_fitLine() +
      # l_rug(alpha=0.5) +
      geom_hline(yintercept = 0, linetype = 2) +
      theme(
        axis.title.x = element_blank(),
        axis.title.y = element_blank()
      ) +
      coord_cartesian(ylim=c(lower, upper)) +
      # coord_cartesian(ylim=c(lower_limits[i], upper_limits[i])) +
      ggtitle(labels[i])
    p$ggObj
  }
  
  cplt = cowplot::plot_grid(plotlist=map(1:length(labels), plt), nrow=3)
  y.grob = textGrob(
    "Smoothed effect (in terms of $$ misallocated)",
    gp=gpar(fontface="bold"),
    rot=90
  )
  plot = grid.arrange(arrangeGrob(cplt, left=y.grob))
  # print(plot)
  
  if (!missing(plotname)) {
    ggsave(sprintf("plots/smooths/%s.png", plotname), plot, width=12, height=6, dpi=300) 
  }
  
  check(viz)
}
