from dp_policy.experiments import Experiment
import click
from typing import Dict

# force inclusion in docs
__pdoc__ = {
    "_run": True,
    "_run_all": True
}


@click.group(chain=True)
def cli():
    pass


@cli.command('run')
@click.argument('name')
@click.option('--just-join', is_flag=True)
@click.option('--no-match-true', is_flag=True)
@click.option('--trials', type=int, default=1000)
def run(*args, **kwargs):
    _run(*args, **kwargs)


def _run(
    name: str,
    just_join: bool = False,
    **kwargs
):
    """Run an experiment and save the results.

    Args:
        name (str): The experiment name.
        just_join (bool, optional): Whether to just join existing results to
            covariates and save the results. Defaults to False.
    """
    print("--- Running", name, "---")
    print("options:", kwargs)
    experiment = Experiment.get_experiment(
        name,
        **kwargs
    )
    if just_join:
        click.echo("Skipping run, straight to join")
        experiment.discrimination_join(include_moes=(name == "baseline"))
    else:
        experiment.run()


@cli.command('run_all')
@click.option('--just-join', is_flag=True)
def run_all(**kwargs):
    experiments = {
        exp: {} for exp in [
            'baseline',
            'hold_harmless',
            'post_processing',
            'thresholds',
            'epsilon',
            'moving_average',
            'budget',
            'sampling',
            'vary_total_children'
        ]
    }
    _run_all(experiments, **kwargs)


def _run_all(experiments: Dict[str, dict], **options):
    """Run multiple experiments.

    Args:
        experiments (Dict[str, dict]): Dictionary of experiment names mapped
            to experiment parameters.
    """
    for name, o in experiments.items():
        o.update(options)
        _run(name, **o)


if __name__ == "__main__":
    cli()
