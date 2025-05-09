# Video Anomaly Detection using Swin Transformer

This project implements an unsupervised video anomaly detection system using a Swin Transformer-based autoencoder. It includes training and prediction scripts adapted for use without distributed training.

## ✅ Requirements
Make sure you have Python 3.10+ and `pip` installed. Then install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🗂️ Dataset Structure
Expected folder layout:

```
dataset/
└── ucsdpeds/
    ├── vidf/         # Contains PNG image frames per scene
    └── labels/       # Optional: contains ground truth labels if available
```

## 🚀 Training the Model
Use `main.py` to train the autoencoder with clustering:

```bash
python main.py --output_dir log_dir --data_path dataset/ucsdpeds/vidf --image_format png
```

For longer training:
```bash
python main.py --output_dir log_dir --data_path dataset/ucsdpeds/vidf --image_format png --epochs 10
```

## 🔍 Running Inference
Use `main_predict.py` to run anomaly detection:

```bash
python main_predict.py --model_pretrain log_dir/checkpoint0.pth --data_path dataset/ucsdpeds/vidf --output_dir ./saving_dir --image_format png

```

## 🧠 Output
- Saves anomaly predictions as PSNR scores
- Calculates AUC (if labels are provided and contain both classes)
- Saves reconstructed videos and logs in `./saving_dir`

## 🧩 Notes
- If you encounter the message `Skipping scene (only one class in labels)`, your ground truth must include at least one anomaly frame (`label == 1`).
- Training checkpoints are saved in `log_dir/`, including `checkpoint0.pth` for inference.

You may expand this list depending on the full output of `pip freeze` from your working environment.

---
