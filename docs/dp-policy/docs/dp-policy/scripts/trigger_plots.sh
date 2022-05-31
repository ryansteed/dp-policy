#!/usr/bin/env bash

# easiest to cp paste this into terminal - then use jobs to monitor
Rscript R/plot_experiment.R baseline 100 > logs/plots_baseline 2>&1 &
Rscript R/plot_experiment.R hold_harmless 100 > logs/plots_hold_harmless 2>&1 &
Rscript R/plot_experiment.R post_processing 100 > logs/plots_post_processing 2>&1 &
Rscript R/plot_experiment.R thresholds 100 > logs/plots_thresholds 2>&1 &
Rscript R/plot_experiment.R moving_average_truth=average 100 > logs/plots_moving_average_truth=average 2>&1 &
Rscript R/plot_experiment.R epsilon 100 > logs/plots_epsilon 2>&1 &
Rscript R/plot_experiment.R budge 100 > logs/plots_budge 2>&1 &
