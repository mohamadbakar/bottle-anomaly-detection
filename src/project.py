from pathlib import Path

import yaml
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    config_path = PROJECT_ROOT / "config" / "patchcore.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    config["data"]["root"] = str((PROJECT_ROOT / config["data"]["root"]).resolve())
    config["engine"]["default_root_dir"] = str(
        (PROJECT_ROOT / config["engine"]["default_root_dir"]).resolve()
    )
    return config


def build_components(config):
    datamodule = MVTecAD(
        root=config["data"]["root"],
        category=config["data"]["category"],
        num_workers=config["data"]["num_workers"],
    )
    model = Patchcore(**config["model"])
    engine = Engine(**config["engine"])
    return datamodule, model, engine
