source("R/utils.R")

trials = 100

for (experiment_name in c(
  "baseline"
  # "hold_harmless",
  # "post_processing",
  # "thresholds",
  # "moving_average_truth=average",
  # "epsilon",
  # "budget"
)) {
  rm(experiment)

  print(experiment_name)
  print("Loading experiment...")
  experiment = load_experiment(experiment_name, trials)
  
  ncols = 3
  if (experiment_name == "budget") {
    ncols = 2
  }
  print("# Race")
  plot_race(experiment, experiment_name, "race_aggregate", ncols)

  # plot_race(experiment, experiment_name, "race", ncols)
  print("# Ethnicity")
  plot_race(experiment, experiment_name, "hispanic", ncols)
  
  # # reduced size for GAM - otherwise takes too long... maybe reverse this later
  # experiment = experiment %>% filter(trial < gam_trials)
  # print(nrow(experiment %>% distinct(trial)))
  # 
  # from_cache = F
  # 
  # for (t in unique(experiment$treatment)) {
  #   print(sprintf("%s: %s", experiment_name, t))
  #   gam_mr = get_gam(
  #     sprintf("%s_%s", experiment_name, t),
  #     F,
  #     from_cache,  # load gam from cache?
  #     experiment %>% filter(treatment == t)
  #   )
  #   plotname = sprintf("%s_%s", experiment_name, t)
  #   viz = get_gam_viz(
  #     plotname,
  #     from_cache,  # load viz from cache?
  #     gam_mr
  #   )
  #   plot_gam(viz, plotname)
  # }
}