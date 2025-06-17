# Video Anomaly Detection using Swin Transformer

> ⚠️ **Important:** The dataset (`ucsdpeds`) is already included in the project under the `dataset/` directory. **You do NOT need to download it separately.**

This project implements an unsupervised video anomaly detection system using a Swin Transformer-based autoencoder. It includes training and prediction scripts adapted for use without distributed training.

## ✅ Requirements
Make sure you have Python 3.10+ and `pip` installed. Then install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🚀 Training the Model
Use `main.py` to train the autoencoder with clustering:

```bash
python main.py --output_dir log_dir --data_path dataset/ucsdpeds/vidf --image_format png
```

## 🔍 Running Inference
Use `main_predict.py` to run anomaly detection:

```bash
python main_predict.py --model_pretrain log_dir/checkpoint0.pth --data_path dataset/ucsdpeds/vidf --output_dir ./saving_dir --image_format png

```
## you can use play in `main_predict` and its will be work.

## 🧠 Output
- Saves anomaly predictions as PSNR scores
- Calculates AUC (if labels are provided and contain both classes)
- Saves reconstructed videos and logs in `./saving_dir`

---
