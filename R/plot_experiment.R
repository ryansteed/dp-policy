source("R/plots.R")

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Two arguments required: experiment name and # of trials.")
}

experiment_name <- args[1]
trials <- as.numeric(args[2])
gam_trials <- trials

print(experiment_name)
print("Loading experiment...")
experiment <- load_experiment(experiment_name, trials)

plot_experiment(experiment)

# reduced size for GAM - otherwise takes too long... maybe reverse this later
experiment <- experiment %>% filter(trial < gam_trials)
print(nrow(experiment %>% distinct(trial)))

gam_experiment(experiment)

print("DONE")
