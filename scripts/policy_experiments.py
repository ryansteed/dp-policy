from dp_policy.experiments import Experiment
import click


@click.group(chain=True)
def cli():
    pass


@cli.command('run')
@click.argument('name')
@click.option('--just-join', is_flag=True)
def run(
    name: str,
    just_join: bool = False,
    **kwargs
):
    print("--- Running", name, "---")
    print("options:", kwargs)
    experiment = Experiment.get_experiment(
        name,
        **kwargs
    )
    if not just_join:
        experiment.run()
    experiment.discrimination_join()


@cli.command('run_all')
@click.option('--just-join', is_flag=True)
def run_all(experiments, **options):
    for name, options in experiments.items():
        run(name, **options)


if __name__ == "__main__":
    cli()
