# Bottle Anomaly Detection using PatchCore

🚀 **[Try the Live Demo](https://bottle-anomaly-detection-f75dmcnsey6aepvcpj4qfc.streamlit.app/)**

An industrial visual inspection application that detects anomalies in bottle images using deep learning and the PatchCore model.

## Project Overview

This project uses the **MVTec AD Bottle dataset** to identify potential defects in bottle images. It provides a web interface where users can upload an image and view the model's prediction and anomaly map.

## Features

- Upload bottle images in PNG or JPG format.
- Detect potential anomalies using PatchCore.
- Display an anomaly score and predicted label.
- Visualize the anomaly map.
- Download the generated anomaly map.
- Run the application through Streamlit.

## Technologies

- Python
- PyTorch
- Anomalib
- PatchCore
- Streamlit
- NumPy
- Matplotlib
- Pillow

## Model Evaluation

The trained model was evaluated on the bottle dataset.

| Metric | Result |
|---|---:|
| Image AUROC | 1.0000 |
| Image F1 Score | 0.9920 |
| Pixel AUROC | 0.9856 |
| Pixel F1 Score | 0.7264 |

These results reflect the evaluation performed in this project. Performance may vary on new images and real-world data.

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/mohamadbakar/bottle-anomaly-detection.git
   cd bottle-anomaly-detection
   ```

2. Create and activate a Python virtual environment.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download or configure access to the trained model checkpoint as described in the project setup.

5. Start the application:

   ```bash
   streamlit run src/app.py
   ```

## Model

The trained PatchCore checkpoint is hosted on Hugging Face:

https://huggingface.co/mohamadbakar/bottle-patchcore-model

## Dataset

This project uses the bottle category from the MVTec Anomaly Detection (MVTec AD) dataset.

## Limitations

The anomaly score is a model score, not a probability. Predictions should be validated on representative production images before using the system for industrial quality control.

## Author

Mohamad Bakar

GitHub: https://github.com/mohamadbakar

