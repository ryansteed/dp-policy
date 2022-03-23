source("R/utils.R")

plot_experiment = function(experiment) {
  ncols = 3
  if (experiment_name == "budget") {
    ncols = 2
  }

  print("- Race")
  plot_race(experiment, experiment_name, "race_aggregate", ncols)
  print("- Ethnicity")
  plot_race(experiment, experiment_name, "hispanic", ncols)
  if (experiment_name == "baseline") {
    print("- Race Detail")
    plot_race(experiment, experiment_name, "race", ncols)
  }
}

gam_experiment = function(experiment, sampling_only) {
  if (missing(sampling_only)) {
    sampling_only = F
  }
  
  print("- GAM")
  from_cache = T
  
  for (t in unique(experiment$treatment)) {
    print(sprintf("%s: %s", experiment_name, t))
    gam_mr = get_gam(
      sprintf("%s_%s", experiment_name, t),
      sampling_only,
      from_cache,  # load gam from cache?
      experiment %>% filter(treatment == t)
    )
    if (sampling_only) {
      plotname = sprintf("%s_%s_sampling", experiment_name, t)
    }
    else {
      plotname = sprintf("%s_%s", experiment_name, t)
    }
    
    gam_table(gam_mr, sprintf("plots/tables/%s.tex", plotname))
    
    viz = get_gam_viz(
      plotname,
      from_cache,  # load viz from cache?
      gam_mr
    )
    plot_gam(viz, plotname)
  }
}
