chooseCRANmirror(ind=1)
# install.packages("pacman")
library(pacman)

pacman::p_load(
  stargazer,
  tidyverse,
  ggpubr,
  janitor,
  mgcv,
  lmtest,
  mgcViz,
  tidymv,
  MASS,
  cowplot,
  grid,
  gridExtra,
  data.table,
  boot,
  broom,
  arrow,
  itsadug,
  xtable,
  comprehenr
)
# p_load_gh(
#   "coolbutuseless/ggpattern"
# )

boot_runs = 1000

clean = function(df) {
  # print(sprintf(
  #   "Dropping %d rows with NA ACS data",
  #   nrow(df %>% clean_names() %>% filter(total_population_race_est == 0) %>% distinct(state_fips_code, district_id))
  # ))
  df_clean = df %>%
    clean_names() %>%
    filter(total_population_race_est != 0) %>%
    mutate_at(
      vars(matches("_pct$")), as.numeric
    ) %>%
    mutate_at(
      vars(matches("race_pct$")), list(`children`= ~(. / 100 * true_children_total))
    ) %>% mutate(
      prop_white = as.numeric(white_race_pct) / 100,
      nonwhite_children = true_children_total * prop_white,
      median_income_est = median_household_income_dollars_income_and_benefits_in_2019_inflation_adjusted_dollars_est,
      pop_density = true_pop_total / aland
    ) %>% mutate(
      # misalloc in terms of true grant total
      misalloc_dp = dp_grant_total - true_grant_total,
      misalloc_sampling = est_grant_total - true_grant_total,
      misalloc_dp_sampling = dpest_grant_total - true_grant_total,
      misalloc_dp_per_child = misalloc_dp / true_children_total,
      misalloc_sampling_per_child = misalloc_sampling / true_children_total,
      misalloc_dp_sampling_per_child = misalloc_dp_sampling / true_children_total
    )
  print(sprintf("%s rows", nrow(df_clean)))
  print(sprintf("%s trials", nrow(df_clean %>% distinct(trial))))
  print(sprintf("%s districts", nrow(df_clean %>% distinct(state_fips_code, district_id))))
  print(
    df_clean %>% 
      summarise_each(funs(sum(is.na(.)))) %>% 
      pivot_longer(everything(), names_to="var", values_to="missing") %>%
      filter(missing > 0)
  )
  return(df_clean)
}

summarise_trials = function(df) {
  grouped = df %>% 
    group_by(treatment, state_fips_code, district_id) %>%
    summarise_at(
      vars(
        misalloc_dp,
        misalloc_sampling,
        misalloc_dp_sampling,
        # all these variables should be constant across trials
        nonwhite_children,
        true_grant_total,
        true_pop_total,
        true_children_eligible,
        true_children_total,
        prop_white,
        ends_with("race_pct"),
        language_other_than_english_language_spoken_at_home_pct,
        foreign_born_place_of_birth_pct,
        not_a_u_s_citizen_u_s_citizenship_status_pct,
        renter_occupied_housing_tenure_pct,
        average_household_size_of_renter_occupied_unit_housing_tenure_est,
        median_income_est,
        pop_density
      ),
      mean
    ) %>%
    ungroup()
  
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

race_comparison_long = function(comparison, kind) {
  if (kind == "hispanic") {
    comparison = comparison %>%
      dplyr::select(ends_with("hispanic_or_latino_and_race_pct") | !ends_with("race_pct"))
  } else if (kind == "race_aggregate") {
    # aggregation of detailed race categories to OMB race minimums based on
    # https://www.socialexplorer.com/data/ACS2017_5yr/metadata/?ds=ACS17_5yr&table=C02003
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
      benefit_dp_sampling = ifelse(race_pct > 0.0, misalloc_dp_sampling_per_child / race_pct * 100, NA),
      children_of_race = round(true_children_total * race_pct / 100),
      children_of_race_eligible = round(true_children_eligible * race_pct / 100)
    )
  
  return(comparison)
}

trim_comparison = function(comparison) {
  return(comparison %>%
    dplyr::select(
      treatment,
      trial,
      ends_with("race_pct"),
      starts_with("true_children"),
      contains("misalloc")
    )
  )
}

summarise_race = function(comparison, kind) {
  comparison %>%
    race_comparison_long(kind) %>%
    group_by(race, trial) %>%
    # compute race-weighted misallocation for each trial
    summarise(
      dp_benefit_per_child = sum(misalloc_dp * race_pct / 100, na.rm=TRUE) / sum(children_of_race, na.rm=TRUE),
      dp_benefit_per_child_eligible = sum(misalloc_dp * race_pct / 100, na.rm=TRUE) / sum(children_of_race_eligible, na.rm=TRUE),
      sampling_benefit_per_child = sum(misalloc_sampling * race_pct / 100, na.rm=TRUE) / sum(children_of_race, na.rm=TRUE),
      sampling_benefit_per_child_eligible = sum(misalloc_sampling * race_pct / 100, na.rm=TRUE) / sum(children_of_race_eligible, na.rm=TRUE),
      dp_sampling_benefit_per_child = sum(misalloc_dp_sampling * race_pct / 100, na.rm=TRUE) / sum(children_of_race, na.rm=TRUE),
      dp_sampling_benefit_per_child_eligible = sum(misalloc_dp_sampling * race_pct / 100, na.rm=TRUE) / sum(children_of_race_eligible, na.rm=TRUE),
      diff_benefit_per_child = dp_sampling_benefit_per_child - sampling_benefit_per_child,
      diff_benefit_per_child_eligible = dp_sampling_benefit_per_child_eligible - sampling_benefit_per_child_eligible
    ) %>%
    ungroup() %>%
    group_by(race) %>%
    # average over trials
    summarise_all(funs(
      mean = mean,
      std_error = sd(.)/sqrt(n())
    )) %>%
    ungroup()
}

race_comparison = function(comparison, kind) {
  comparison =  comparison %>%
    trim_comparison() %>%
    group_by(treatment) %>%
    # group modify to be more memory efficient (but slower) - otherwise longified df is too long
    group_modify(~ summarise_race(.x, kind))

  # comparison_mean = comparison %>%
  #   group_by(treatment, race) %>%
  #   summarise(
  #     dp_benefit_per_child = sum(misalloc_dp * race_pct / 100) / sum(children_of_race),
  #     dp_benefit_per_child_eligible = sum(misalloc_dp * race_pct / 100) / sum(children_of_race_eligible),
  #     sampling_benefit_per_child = sum(misalloc_sampling * race_pct / 100) / sum(children_of_race),
  #     sampling_benefit_per_child_eligible = sum(misalloc_sampling * race_pct / 100) / sum(children_of_race_eligible),
  #     dp_sampling_benefit_per_child = sum(misalloc_dp_sampling * race_pct / 100) / sum(children_of_race),
  #     dp_sampling_benefit_per_child_eligible = sum(misalloc_dp_sampling * race_pct / 100) / sum(children_of_race_eligible)
  #   )
  
  sorted = comparison %>%
    mutate(race = fct_reorder(race, sampling_benefit_per_child_eligible_mean)) %>%
    distinct(
      treatment, race, 
      sampling_benefit_per_child_eligible_mean,
      sampling_benefit_per_child_eligible_std_error,
      dp_sampling_benefit_per_child_eligible_mean,
      dp_sampling_benefit_per_child_eligible_std_error,
      diff_benefit_per_child_eligible_mean,
      diff_benefit_per_child_eligible_std_error
    )
  
  return(sorted)
}

load_experiment = function(name, max_trials) {
  # raw = fread(sprintf("results/policy_experiments/%s_discrimination_laplace.csv", name))
  raw = read_feather(sprintf("results/policy_experiments/%s_discrimination_laplace.feather", name))
  if (!missing(max_trials)) {
    print(sprintf("Limiting to %d trials", max_trials))
    raw = raw %>% filter(trial < max_trials)
  }
  print("Cleaning...")
  df = clean(raw)
  return(df)
}


# plot_race_bar = function(comparison, ncol) {
#   plt = ggplot(comparison, aes(x=race, y=sampling_benefit_per_child_eligible_mean)) +
#     geom_col(position="dodge", aes(fill=treatment), color="black") +
#     geom_errorbar(
#       aes(
#         # x=race,
#         # y=dp_sampling_benefit_per_child,
#         ymin=sampling_benefit_per_child_eligible_mean, 
#         ymax=dp_sampling_benefit_per_child_eligible_mean,
#         # linetype="",
#         color="",
#         fill=treatment
#       ),
#       position=position_dodge(width=0.9),
#       size=0.5,
#       width=0.5
#     ) +
#     ylab("Race-weighted misallocation per eligible child") +
#     scale_colour_manual(values=c("black")) +
#     coord_flip() +
#     xlab("Census Race Category") +
#     guides(
#       fill = guide_legend(ncol=ncol)
#     ) +
#     labs(
#       fill = "Data error",
#       color = "+ DP error"
#     ) +
#     theme(
#       legend.position = "top",
#       legend.box="vertical"
#     )
#   return(plt)
# }
# 
# 
# plot_race_lollipop = function(comparison, ncol) {
#   # comparison_long = comparison %>%
#   #   pivot_longer(c(sampling_benefit_per_child_eligible, dp_sampling_benefit_per_child_eligible))
#   
#   dodge = position_dodge(width=0.5)
#   
#   plt = ggplot(comparison, aes(x=race, y=sampling_benefit_per_child_eligible_mean)) +
#     # geom_col(position="dodge", aes(fill=treatment), color="black") +
#     geom_linerange(
#       aes(
#         ymin=sampling_benefit_per_child_eligible_mean, 
#         ymax=dp_sampling_benefit_per_child_eligible_mean,
#         fill=treatment
#       ),
#       color="black",
#       position=dodge,
#       size=0.5
#     ) +
#     geom_point(
#       aes(
#         y=dp_sampling_benefit_per_child_eligible_mean,
#         fill=treatment
#       ),
#       color="black",
#       size=1.5,
#       position=dodge
#     ) +
#     geom_linerange(
#       aes(
#         ymin=0, 
#         ymax=sampling_benefit_per_child_eligible_mean,
#         color=treatment
#       ),
#       position=dodge,
#       size=1
#     ) +
#     geom_point(
#       aes(
#         y=sampling_benefit_per_child_eligible_mean,
#         color=treatment
#       ),
#       size=3,
#       position=dodge
#     ) +
#     ylab("Race-weighted misallocation per eligible child") +
#     # scale_colour_manual(values=c("black")) +
#     coord_flip() +
#     xlab("Census Race Category") +
#     guides(
#       fill = guide_legend(ncol=ncol)
#     ) +
#     labs(
#       fill = "Data error",
#       color = "+ DP error"
#     ) +
#     theme(
#       legend.position = "top",
#       legend.box="vertical"
#     )
#   return(plt)
# }

plot_race_bar_stacked = function(comparison, ncol, label_width, alpha) {
  if (missing(alpha)) {
    alpha = 0.01
  }
  alpha_sig = alpha
  comparison = comparison %>%
    ungroup() %>%
    mutate(
      dp_moe = qnorm(1-alpha/2) * dp_sampling_benefit_per_child_eligible_std_error,
      sampling_moe = qnorm(1-alpha/2) * sampling_benefit_per_child_eligible_std_error,
      diff_moe = qnorm(1-alpha_sig/2) * diff_benefit_per_child_eligible_std_error,
      sigdiff = ifelse((
        ((diff_benefit_per_child_eligible_mean - diff_moe) < 0) &
          ((diff_benefit_per_child_eligible_mean + diff_moe) > 0)
      ), "notsig", "sig"),
      treatment = fct_reorder(
        as.factor(treatment), as.integer(str_detect(treatment, "baseline")),
      )
    )
  include_sig = any(comparison$sigdiff == "notsig")

  # for baseline
  if (nrow(comparison %>% distinct(treatment)) == 1) {
    print("Just printing one treatment")
    comparison$treatment = as.factor("")
  }
  
  # specify default value - used for baseline
  # then use that color anytime "baseline" appears in treatment name
  # otherwise, use a normal palette
  palette = function(treatments) {
    pal = RColorBrewer::brewer.pal(8, "Accent")
    default_color = tail(pal, n=1)
    pal = head(pal, n=7)
    setNames(ifelse(grepl("baseline", treatments), default_color, pal), treatments)
  }
  
  plt = ggplot(comparison, aes(x=race, y=sampling_benefit_per_child_eligible_mean)) +
    geom_col(
      position="dodge",
      aes(fill=treatment),
    ) +
    geom_col(
      aes(
        y=dp_sampling_benefit_per_child_eligible_mean,
        shape=treatment,
        linetype=sigdiff
      ),
      color="black",
      fill="transparent",
      position="dodge"
    ) +
    # dp errorbar
    # geom_errorbar(
    #   aes(
    #     y = dp_sampling_benefit_per_child_eligible_mean,
    #     ymin=(
    #       dp_sampling_benefit_per_child_eligible_mean - dp_moe
    #     ), 
    #     ymax=(
    #       dp_sampling_benefit_per_child_eligible_mean + dp_moe
    #     ),
    #     # linetype="",
    #     fill=treatment
    #   ),
    #   color="black",
    #   position=position_dodge2(width=0.9),
    #   size=0.5,
    #   width=0.5
    # ) +
    # sampling errorbar
    geom_errorbar(
      aes(
        y = sampling_benefit_per_child_eligible_mean,
        ymin=(
          sampling_benefit_per_child_eligible_mean - dp_moe
        ), 
        ymax=(
          sampling_benefit_per_child_eligible_mean + dp_moe
        ),
        # linetype="",
        fill=treatment
      ),
      color="black",
      position=position_dodge(width=0.9),
      size=0.5,
      width=0.5
    ) +
    geom_point(
      aes(
        y = sampling_benefit_per_child_eligible_mean,
        fill = treatment,
        shape = treatment
      ),
      colour="black",
      size=1.5,
      stroke=0.75,
      position=position_dodge(width=0.9)
    ) +
    ylab("Race-weighted misallocation per eligible child ($)")
  if (include_sig) {
    plt = plt + scale_linetype_manual(
        values=c("sig" = "solid", "notsig" = "dashed"),
        labels=c("sig" = sprintf("Significantly\ndifferent\n(p<%.2f)", alpha_sig), "notsig" = "Not\nsignificant")
      )
  } else {
    print("Ignoring insig")
    plt = plt + scale_linetype_manual(
        values=c("sig" = "solid"),
        labels=c("sig" = sprintf("Significantly\ndifferent\n(p<%.2f)", alpha_sig))
      )
  }

  plt = plt +
    scale_fill_manual(
      labels = function(x) str_wrap(x, width=5),
      values = palette(levels(comparison$treatment)),
      guide = guide_legend(ncol=ncol, order=1, reverse=TRUE)
    ) +
    scale_shape_manual(
      labels = function(x) str_wrap(x, width=5),
      values = c(21, 22, 23, 24, 25, 3, 4),
      guide = guide_legend(ncol=ncol, order=1, reverse=TRUE)
    ) +
    scale_x_discrete(
      labels = function(x) str_wrap(x, width=label_width)
    ) +
    scale_y_continuous(
      breaks = scales::breaks_pretty(8)
    ) +
    coord_flip() +
    xlab("Census Race Category") +
    guides(
      linetype = guide_legend(ncol=2, order=2)
    ) +
    labs(
      fill = "Data\ndeviations",
      shape = "Data\ndeviations",
      linetype = "+ privacy\ndevations\n(ε=0.1)"
    ) +
    theme(
      legend.position = "top",
      legend.box=ifelse(include_sig, "vertical", "horizontal"),
      axis.ticks = element_blank()
    )
  
  return(plt)
}

plot_ru_by_race = function(comparison, marginal, alpha) {
  if (missing(alpha)) {
    alpha = 0.01
  }
  alpha_sig = alpha

  comparison = comparison %>%
    mutate(
      avg = if(marginal) {
        diff_benefit_per_child_eligible_mean
      } else {
        dp_sampling_benefit_per_child_eligible_mean
      },
      std_error = if(marginal) {
        diff_benefit_per_child_eligible_std_error
      } else {
        dp_sampling_benefit_per_child_eligible_std_error
      },
      moe = qnorm(1 - alpha / 2) * std_error
    )

  plt = ggplot(comparison, aes(x=treatment, y=avg)) +
    geom_errorbar(aes(ymin=avg - moe, ymax = avg + moe, color=race), width=0.1) +
    geom_line(aes(color=race)) +
    geom_point(aes(color=race, shape=race)) +
    scale_x_continuous(trans='log10') +
    scale_color_brewer(palette="Accent") +
    xlab("Privacy parameter ε") +
    ylab(ifelse(
      marginal,
      "Marginal race-weighted misallocation per eligible child",
      "Race-weighted misallocation per eligible child"
    )) +
    labs(
      color = "",
      shape = ""
    ) +
    guides(
      color = guide_legend(ncol=3),
      shape = guide_legend(ncol=3)
    ) +
    theme(
      legend.position = "top",
      legend.box="vertical"
    )
  
  return(plt)
}

plot_race = function(name, trials, kind, from_cache, ncol) {
  if (missing(ncol)) {
    ncol = 3
  }
  kind_formatted = sprintf("_%s", kind)
  if (kind == "race_aggregate") {
    kind_formatted = ""
  }
  
  # no longer grouping before computation...
  # print("Grouping...")
  # grouped = summarise_trials(experiment)
  
  print("Comparing...")
  savepath = sprintf(
    "results/policy_experiments/%s_comparison%s_trials=%d.rds",
    name,
    kind_formatted,
    trials
  )
  if (file.exists(savepath) & from_cache) {
    print("Reading comparison from cache...")
    comparison = readRDS(savepath)
  } else {
    print("Generating comparison from scratch...")
    comparison = race_comparison(load_experiment(name, trials), kind)
    saveRDS(comparison, savepath)
  }
  
  print("Plotting...")
  if (name == "epsilon") {
    print("Also plotting R-U curves")
    # for epsilon experiment, disparities are too large to see if we include 0.001
    comparison = comparison %>% filter(treatment > 0.001)
    plt = plot_ru_by_race(comparison, FALSE)
    ggsave(sprintf("plots/race/ru_%s%s.pdf", name, kind_formatted), dpi=300, width=6, height=7.2, bg='transparent', device=cairo_pdf)
    plt_marginal = plot_ru_by_race(comparison, TRUE)
    ggsave(sprintf("plots/race/ru_marginal_%s%s.pdf", name, kind_formatted), dpi=300, width=6, height=7.2, bg='transparent', device=cairo_pdf)
  }

  plt = plot_race_bar_stacked(comparison, ncol, ifelse(kind == "race", 16, 8))
  print(plt)
  ggsave(sprintf("plots/race/misalloc_%s%s.pdf", name, kind_formatted), dpi=300, width=6, height=7.2, bg='transparent', device=cairo_pdf)
}

sq_share = function(x) {
  return((x/100)^2)
}

clean_for_reg = function(df, sampling_only) {
  if (missing(sampling_only)) {
    sampling_only = F
  }
  
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
      prop_hispanic = 1-not_hispanic_or_latino_hispanic_or_latino_and_race_pct/100,
    ) %>%
    # impute mean
    mutate_at(
      c("median_income_est", "average_household_size_of_renter_occupied_unit_housing_tenure_est"),
      ~ replace(.x, is.na(.x), median(.x))
    ) %>%
    dplyr::select(
      -ends_with("hispanic_or_latino_and_race_pct"),
      -asian_race_pct
    )
  # filter(trial < 10)
  if (sampling_only) {
    df = df %>% mutate(misalloc = misalloc_sampling)
  } else {
    df = df %>% mutate(misalloc = misalloc_dp_sampling)
  }
  return(df)
}

run_regs = function(df, sampling_only, savepath) {
  df_reg = clean_for_reg(df, sampling_only)
  
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
    data=df_reg
  )
  print(summary(gam_mr))
  
  if (!missing(savepath)) {
    saveRDS(gam_mr, savepath)
  }
  
  return(gam_mr)
}

gamtabs_summary = function(gam, ...) {
  s = summary(gam)
  gamtabs(gam, caption=sprintf(
    "R-sq. (adj) = %.4f, Deviance explained = %.2f%%, -REML = %.4e, Scale est. = %.4e, n = %d",
    s$r.sq,
    s$dev.expl*100,
    s$sp.criterion["REML"],
    s$scale,
    s$n
  ), ...)
}

regression_tables = function(experiment_name, sampling_only, trials) {
  if (missing(trials)) {
    trials = 100
  }
  
  experiment = load_experiment(experiment_name) %>% filter(trial < trials)
  
  df_reg = clean_for_reg(experiment, sampling_only)
  
  print("- OLS")
  lm_formula = lm(
    misalloc ~
      log(true_pop_total) +
      true_children_total +
      true_children_poverty,
    data=df_reg
  )
  lm_final = lm(
    misalloc ~
      log(pop_density) +
      hhi +
      prop_white +
      prop_hispanic +
      median_income_est +
      renter_occupied_housing_tenure_pct,
    data=df_reg
  )
  lm_extra = lm(
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
    data=df_reg
  )
  stargazer(
    lm_formula, lm_extra, lm_final,
    type="latex",
    column.labels = c("Formula components", "All variables", "Demographic variables"),
    dep.var.labels = "Misallocation (deviated minus true)",
    covariate.labels=c(
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
    out=sprintf("plots/tables/%s_ols.tex", experiment_name)
  )
  
  print("- GAM")
  gam_mr = get_gam("baseline", sampling_only, TRUE, experiment)
  gam_table(gam_mr, sprintf("plots/tables/%s_gam.tex", experiment_name))
  print(
    xtable(anova(lm_final, gam_mr)),
    file=sprintf("plots/tables/%s_anova_lm_gam.tex", experiment_name)
  )
  
  print("- GAM with interaction")
  gam_mr_interact = gam(
    misalloc ~
      te(
        prop_white,
        median_income_est
      ) +
      s(log(pop_density), bs="tp") +
      s(hhi, bs="tp") +
      s(prop_hispanic, bs="tp") +
      s(renter_occupied_housing_tenure_pct, bs="tp"),
    data=df_reg
  )
  sink(sprintf("plots/tables/%s_interact.tex", experiment_name))
  gamtabs_summary(gam_mr_interact, label="Demographic GAN with interactions")
  sink()
  stargazer(
    anova.gam(gam_mr, gam_mr_interact, test="F"),
    type="latex",
    summary=FALSE,
    out=sprintf("plots/tables/%s_anova_gam_interact.tex", experiment_name)
  )
}

gam_table = function(gam, savepath) {
  sink(savepath)
  gamtabs_summary(
    gam,
    label="Demographic GAM",
    snames=c(
      "Log population density",
      "Racial homogeneity (HHI)",
      "Proportion White",
      "Proportion Hispanic",
      "Median income",
      "% households renting"
    )
  )
  sink()
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

sanitize = function(string) {
  return(str_replace(string, "%", ""))
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
    ggsave(
      sprintf("plots/smooths/%s.pdf", sanitize(plotname)),
      plot, width=12, height=6, dpi=300, bg='transparent'
    )
  }
  
  check(viz)
}
