"""
Command-line interface for the multimodal_fin package.
Defines the Typer app and entry points for both console_scripts and direct execution.
"""
from pathlib import Path
import typer

from multimodal_fin.config import load_settings
from multimodal_fin.processors.conference import ConferenceProcessor

app = typer.Typer(help="Multimodal conference processing CLI.")


@app.command()
def main(
    config_file: Path = typer.Option(..., help="Path to the YAML configuration file."),
    config_name: str = typer.Option("default", help="Configuration section name in the YAML file."),
) -> None:
    """
    Execute the full pipeline: preprocessing, classification, and multimodal analysis.

    Args:
        config_file (Path): Path to the YAML file containing pipeline settings.
        config_name (str): Key within the YAML under 'configs' indicating which settings to use.

    Returns:
        None
    """
    # Load and validate settings
    settings = load_settings(str(config_file), config_name)

    # Instantiate and run the main conference processor
    processor = ConferenceProcessor(settings)
    processor.run()


def cli() -> None:
    """
    Console-script entrypoint: invokes the Typer app.
    """
    app()


if __name__ == "__main__":
    cli()