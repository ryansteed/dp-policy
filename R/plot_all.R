source("R/plots.R")

trials = 100
gam_trials = trials

for (experiment_name in c(
  # "baseline",
  # "hold_harmless",
  # "post_processing",
  # "thresholds",
  # "moving_average_truth=average",
  # "budget",
  # "epsilon",
  "sampling"
)) {
  print("\n")
  print(experiment_name)
  print("Loading experiment...")
  experiment = load_experiment(experiment_name, trials)

  plot_experiment(experiment)
  
  # reduced size for GAM - otherwise takes too long... maybe reverse this later
  experiment = experiment %>% filter(trial < gam_trials)
  print(nrow(experiment %>% distinct(trial)))

  gam_experiment(experiment)
  if (experiment_name == "baseline") {
    print("Also running sampling alone")
    gam_experiment(experiment, T)
  }

  print(sprintf("DONE with %s", experiment_name))
  rm(experiment)
}
