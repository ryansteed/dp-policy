#!/usr/bin/env bash

# easiest to cp paste this into terminal - then use jobs to monitor
dp_policy run hold_harmless > logs/hold_harmless 2>&1 & \
dp_policy run post_processing > logs/post_processing 2>&1 & \
dp_policy run thresholds > logs/thresholds 2>&1 & \
dp_policy run moving_average > logs/moving_average 2>&1 & \
dp_policy run budget > logs/budget 2>&1 & \
dp_policy run epsilon > logs/epsilon 2>&1 & \
dp_policy run sampling > logs/sampling 2>&1 &
