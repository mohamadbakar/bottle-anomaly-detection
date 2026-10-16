# Bottle anomaly detection with PatchCore

This project trains and evaluates PatchCore on the MVTec AD `bottle` category.
The dataset is read from `datasets/bottle`; the scripts do not extract, move,
or modify dataset files.

## Setup

Use a Python environment compatible with Anomalib 2.7.0, then install the
declared dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train

Run from the project root:

```powershell
python src\train.py
```

Training outputs are written under `results`.

## Evaluate

Pass the checkpoint produced by training:

```powershell
python src\evaluate.py results\path\to\model.ckpt
```

The data path, category, PatchCore settings, and engine settings are in
`config\patchcore.yaml`.
