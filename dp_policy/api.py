from dp_policy.experiments import Experiment
import click


@click.group(chain=True)
def cli():
    pass


@cli.command('run')
@click.argument('name')
@click.option('--just-join', is_flag=True)
@click.option('--trials', type=int, default=1000)
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
    if just_join:
        click.echo("Skipping run, straight to join")
    else:
        experiment.run()


@cli.command('run_all')
@click.option('--just-join', is_flag=True)
def run_all(experiments, **options):
    for name, options in experiments.items():
        run(name, **options)


if __name__ == "__main__":
    cli()
