# Deepfake Detection: Model Training and Evaluation

ResNet18 (ImageNet-pretrained) fine-tuned to classify face images as **real** or **fake**.

## Results (test set, 2,000 images)

| Metric | Value |
|---|---|
| Accuracy | 0.9075 |
| Precision | 0.9031 |
| Recall | 0.9130 |
| F1 | 0.9080 |
| ROC-AUC | 0.9641 |

## Setup

- Data: subset of the 140k Real and Fake Faces dataset (5,000 + 5,000 train, 1,000 + 1,000 valid, 1,000 + 1,000 test)
- Model: ResNet18, final layer replaced by `Linear(512, 1)`
- Loss and optimizer: BCEWithLogitsLoss, Adam (lr 1e-4)
- Training: 4 epochs, batch size 64
- Input: 128x128, ImageNet normalization
- Class mapping: **fake = 0, real = 1** (the model outputs the probability of real)

## Files

```
code/
  config.py       shared constants
  train.py        training
  evaluate.py     test evaluation and figures
results/
  metrics.json
  history.json
  confusion_matrix.png
  roc_curve.png
  training_curves.png
  misclassified.png
```

## How to run

```
pip install torch torchvision numpy matplotlib scikit-learn pillow
cd code
mkdir -p ../out/models ../out/results
python train.py --data path/to/data --out ../out
python evaluate.py --data path/to/data --out ../out
```

The data folder must contain `train/`, `valid/` and `test/`, each with `fake/` and `real/` subfolders.
