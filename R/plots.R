source("R/utils.R")

trials = 100
gam_trials = trials

plot_experiment = function(experiment) {
  ncols = 3
  if (experiment_name == "budget") {
    ncols = 2
  }

  print("- Race")
  plot_race(experiment, experiment_name, "race_aggregate", ncols)
  print("- Race Detail")
  plot_race(experiment, experiment_name, "race", ncols)
  print("- Ethnicity")
  plot_race(experiment, experiment_name, "hispanic", ncols)
}

gam_experiment = function(experiment) {
  print("- GAM")
  from_cache = T
  
  for (t in unique(experiment$treatment)) {
    print(sprintf("%s: %s", experiment_name, t))
    gam_mr = get_gam(
      sprintf("%s_%s", experiment_name, t),
      F,
      from_cache,  # load gam from cache?
      experiment %>% filter(treatment == t)
    )
    plotname = sprintf("%s_%s", experiment_name, t)
    viz = get_gam_viz(
      plotname,
      from_cache,  # load viz from cache?
      gam_mr
    )
    plot_gam(viz, plotname)
  }
}
