import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from project import PROJECT_ROOT, build_components, load_config

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "Patchcore"
    / "MVTecAD"
    / "bottle"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)


def main():
    parser = argparse.ArgumentParser(description="Predict anomalies in one bottle image.")
    parser.add_argument("image", type=Path, help="Image path, for example C:\\images\\bottle.png")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--save-map",
        type=Path,
        help="Save the anomaly map to this path instead of displaying it.",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    config = load_config()
    config["model"]["pre_trained"] = False
    _, model, engine = build_components(config)

    predictions = engine.predict(
        model=model,
        data_path=image_path,
        ckpt_path=checkpoint_path,
        return_predictions=True,
    )
    if not predictions:
        raise RuntimeError("Anomalib returned no prediction for the image.")

    prediction = predictions[0]
    if prediction.pred_score is None:
        raise RuntimeError("Anomalib did not return an anomaly score.")
    score = prediction.pred_score.reshape(-1)[0].detach().cpu().item()
    print(f"Anomaly score: {score:.6f}")

    if prediction.pred_label is not None:
        label = prediction.pred_label.reshape(-1)[0].detach().cpu().item()
        print(f"Anomalib prediction label: {label}")
    else:
        print("Anomalib prediction label: unavailable; no custom threshold was applied.")

    if prediction.anomaly_map is None:
        print("Anomaly map: unavailable.")
        return

    anomaly_map = prediction.anomaly_map.squeeze().detach().cpu().numpy()
    figure, axis = plt.subplots()
    plot = axis.imshow(anomaly_map, cmap="inferno")
    axis.set_title("PatchCore anomaly map")
    axis.axis("off")
    figure.colorbar(plot, ax=axis, label="Anomaly score")
    figure.tight_layout()

    if args.save_map:
        output_path = args.save_map.resolve()
        if output_path.exists():
            parser.error(f"Refusing to overwrite existing file: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"Anomaly map saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
