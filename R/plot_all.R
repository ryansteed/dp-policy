source("R/plots.R")

for (experiment_name in c(
  # "baseline",
  # "hold_harmless",
  # "post_processing",
  "thresholds",
  "moving_average_truth=average",
  "epsilon",
  "budget"
)) {
  rm(experiment)

  print(experiment_name)
  print("Loading experiment...")
  experiment = load_experiment(experiment_name, trials)

  # plot_experiment(experiment)
  
  # reduced size for GAM - otherwise takes too long... maybe reverse this later
  experiment = experiment %>% filter(trial < gam_trials)
  print(nrow(experiment %>% distinct(trial)))

  gam_experiment(experiment)
}
