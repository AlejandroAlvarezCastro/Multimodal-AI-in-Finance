import os
import yaml
import argparse
from MULTIMODAL.ConferenceProcessor import load_config

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    parser = argparse.ArgumentParser(description="Procesador de conferencias multimodales")
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Nombre de la configuración a usar (debe estar bajo 'configs' en config.yaml)"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Ruta al archivo YAML con las configuraciones (por defecto: config.yaml)"
    )
    args = parser.parse_args()

    # Leer archivo de configuración
    with open(args.config_file, "r") as f:
        data = yaml.safe_load(f)

    if "configs" not in data:
        raise ValueError("No se encontró la clave 'configs' en el archivo YAML.")

    available_configs = list(data["configs"].keys())

    if args.config_name not in available_configs:
        raise ValueError(f"Configuración '{args.config_name}' no encontrada. Disponibles: {available_configs}")

    clear_terminal()

    processor = load_config(args.config_file, config_name=args.config_name)
    processor.run()

if __name__ == "__main__":
    main()
