from dp_policy.experiments import Experiment


def run(experiments, **kwargs):
    for name, options in experiments.items():
        print("--- Running", name, "---")
        experiment = Experiment.get_experiment(
            name,
            **kwargs
        )
        if not options.get('join_only'):
            experiment.run()
        experiment.discrimination_join()


if __name__ == "__main__":
    run({
        # "hold_harmless": {},
        # "post_processing": {},
        # "moving_average_truth=average": {},
        "thresholds": {},
        "epsilon": {},
        "budget": {}
    })
