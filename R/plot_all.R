source("R/plots.R")

trials_main = 1000
trials_appendix = 1000
gam_trials = 100

for (experiment_name in c(
  # "baseline",
  # "hold_harmless",
  # "post_processing",
  # "thresholds",
  # "moving_average_truth=average",
  # "budget",
  # "epsilon",
  # "sampling",
  "vary_total_children",
  "hold_harmless_unmatched"
)) {
  print("\n")
  print(experiment_name)

  plot_experiment(
    experiment_name,
    if (experiment_name %in% c("baseline", "hold_harmless")) trials_main else trials_appendix
  )
  # reduced size for GAM - otherwise takes too long... maybe reverse this later
  # experiment = load_experiment(experiment_name, gam_trials)
  # print(nrow(experiment %>% distinct(trial)))

  # gam_experiment(experiment)
  # if (experiment_name == "baseline") {
  #   print("Also running sampling alone")
  #   gam_experiment(experiment, T)
  # }

  print(sprintf("DONE with %s", experiment_name))
  rm(experiment)
}
