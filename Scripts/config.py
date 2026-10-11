"""Lit config.yaml et le donne aux autres fichiers sous forme d'un dictionnaire `cfg`."""
from pathlib import Path

import yaml

# config.yaml est dans le dossier PARENT de Scripts : on peut donc lancer
# les scripts depuis n'importe quel dossier.
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(config_path=DEFAULT_CONFIG):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml introuvable : {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Vérification simple : le seuil du masque doit être entre 0 et 1.
    if not 0 < cfg["mask"]["threshold_fraction"] < 1:
        raise ValueError("mask.threshold_fraction doit être entre 0 et 1")

    # On transforme les chemins relatifs en chemins complets.
    root = config_path.resolve().parent
    cfg["paths"] = {nom: root / chemin for nom, chemin in cfg["paths"].items()}
    return cfg