source("R/utils.R")

plot_experiment <- function(experiment_name, trials) {
  #' Plots race disparity plots for an experiment using a given number of
  #' trials.
  ncols <- 3
  if (experiment_name == "budget") {
    ncols <- 2
  }

  from_cache <- TRUE

  print("- Race")
  plot_race(experiment_name, trials, "race_aggregate", from_cache, ncols)
  print("- Ethnicity")
  plot_race(experiment_name, trials, "hispanic", from_cache, ncols)
  if (experiment_name %in% c(
    "baseline",
    "hold_harmless",
    "hold_harmless_unmatched",
    "vary_total_children"
  )) {
    print("- Race Detail")
    plot_race(experiment_name, trials, "race", from_cache, ncols)
  }
}

gam_experiment <- function(experiment_name, trials, sampling_only) {
  #' Run GAM and plot smooths for a given experiment.
  #' If `sampling_only`, show the effects for only data deviations.
  #' Otherwise, show effects for combined privacy and data deviations.

  if (missing(sampling_only)) {
    sampling_only <- FALSE
  }

  print("- GAM")
  
  from_cache <- TRUE
  
  if (from_cache) {
    # infer treatments from filenames
    experiment = NULL
    treatments = list.files(
      path = "results/regressions",
      sprintf("^%s_*._sampling=%s", experiment_name, sampling_only)
    ) %>%
      str_remove(".rds") %>%
      str_remove("sampling=TRUE|sampling=FALSE") %>%
      str_remove(sprintf("%s_", experiment_name))
    if (experiment_name == "hold_harmless") {
      treatments = treatments %>% filter(grepl("unmatched", treatments))
    }
    if (experiment_name == "baseline") {
      treatments = c("baseline")
    }
  } else {
    experiment = load_experiment(experiment_name, trials)
    treatments = unique(experiment$treatment)
  }
  
  for (t in treatments) {
    print(t)
    
    print(sprintf("%s: %s", experiment_name, t))
    gam_mr <- get_gam(
      sprintf("%s_%s", experiment_name, t),
      sampling_only,
      from_cache,  # load gam from cache?
      experiment %>% filter(treatment == t)
    )
    if (sampling_only) {
      plotname <- sprintf("%s_%s_sampling", experiment_name, t)
    } else {
      plotname <- sprintf("%s_%s", experiment_name, t)
    }
    gam_table(gam_mr, sprintf("plots/tables/%s.tex", plotname))

    viz <- get_gam_viz(
      plotname,
      from_cache,  # load viz from cache?
      gam_mr
    )
    plot_gam(viz, plotname)
  }
}
