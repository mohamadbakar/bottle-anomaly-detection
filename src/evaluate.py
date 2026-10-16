import argparse
from pathlib import Path

from project import build_components, load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained bottle PatchCore model.")
    parser.add_argument("checkpoint", type=Path, help="Path to the trained .ckpt file")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    config = load_config()
    datamodule, model, engine = build_components(config)
    engine.test(model=model, datamodule=datamodule, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
