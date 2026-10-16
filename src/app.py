import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from project import PROJECT_ROOT, load_config

CHECKPOINT_PATH = (
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


@st.cache_resource(show_spinner=False)
def load_predictor():
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    config = load_config()
    model = Patchcore.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location="cpu",
        pre_trained=False,
    )
    engine = Engine(**config["engine"])
    return model, engine


def make_anomaly_map_png(anomaly_map: np.ndarray) -> bytes:
    figure, axis = plt.subplots()
    image = axis.imshow(anomaly_map, cmap="inferno")
    axis.set_title("PatchCore anomaly map")
    axis.axis("off")
    figure.colorbar(image, ax=axis, label="Anomaly score")
    figure.tight_layout()

    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


st.set_page_config(page_title="Bottle anomaly inspection", page_icon="🔎")
st.title("Bottle anomaly inspection")
st.write("Upload one bottle image to inspect it with the trained PatchCore model.")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=("png", "jpg", "jpeg"),
)

if uploaded_image is not None:
    image_bytes = uploaded_image.getvalue()
    try:
        with Image.open(io.BytesIO(image_bytes)) as image_file:
            image_format = image_file.format
            image_file.verify()
        if image_format not in {"PNG", "JPEG"}:
            raise ValueError(f"Unsupported image format: {image_format}")
        with Image.open(io.BytesIO(image_bytes)) as image_file:
            display_image = image_file.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"The uploaded file is not a readable PNG or JPG image: {exc}")
        st.stop()

    st.image(display_image, caption=uploaded_image.name, use_column_width=True)

    temporary_path = None
    try:
        with st.spinner("Loading the model and analyzing the image..."):
            model, engine = load_predictor()
            suffix = ".png" if image_format == "PNG" else ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                temporary_file.write(image_bytes)
                temporary_path = Path(temporary_file.name)

            predictions = engine.predict(
                model=model,
                data_path=temporary_path,
                return_predictions=True,
            )

        if not predictions:
            st.error("Anomalib returned no prediction for this image.")
        else:
            prediction = predictions[0]
            if prediction.pred_score is None:
                st.error("Anomalib did not return an anomaly score.")
            else:
                score = prediction.pred_score.reshape(-1)[0].detach().cpu().item()
                st.metric("Anomaly score (not a probability)", f"{score:.6f}")

            if prediction.pred_label is None:
                st.info("Anomalib did not return a True/False label for this image.")
            else:
                label = bool(prediction.pred_label.reshape(-1)[0].detach().cpu().item())
                st.write(f"Anomalib label: **{label}**")

            if prediction.anomaly_map is None:
                st.info("No anomaly map was returned for this image.")
            else:
                anomaly_map = prediction.anomaly_map.squeeze().detach().cpu().numpy()
                map_png = make_anomaly_map_png(anomaly_map)
                st.image(map_png, caption="Anomaly map", use_column_width=True)
                st.download_button(
                    "Download anomaly map",
                    data=map_png,
                    file_name=f"{Path(uploaded_image.name).stem}_anomaly_map.png",
                    mime="image/png",
                )
    except Exception as exc:
        st.error(f"Could not load the model or analyze this image: {exc}")
        st.exception(exc)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
