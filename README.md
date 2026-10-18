AI Skin Disease Detection

Overview
- Transfer-learning-based classifier for skin disease images.

Quick start
1. Add images: place images into `dataset/<disease-name>/` directories. Each class must be a separate folder.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Train a model (creates `skin_model_transfer.keras` and `classes.json`):

```bash
python train.py
```

4. Run the web app:

```bash
streamlit run app.py
```

Notes & tips
- The training script uses MobileNetV2 transfer learning. For reliable results you should provide at least dozens of images per class.
- To add more diseases, create new folders under `dataset/` (for example `dataset/acne/`, `dataset/eczema/`, `dataset/psoriasis/`, `dataset/new_disease/`) and add images.
- After adding images, re-run `python train.py` to update the model and `classes.json`.
- If you don't have many images, consider collecting more data or using data-augmentation and cross-validation strategies.

Troubleshooting
- If the app shows the same disease for all images, the most common reason is insufficient and/or imbalanced data. Collect more images per class and retrain.
- If `train.py` fails due to missing packages, install TensorFlow and dependencies from `requirements.txt`.
