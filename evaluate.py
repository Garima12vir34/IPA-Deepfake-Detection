import argparse, json
import numpy as np, torch, torch.nn as nn, torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from config import IMG_SIZE, MEAN, STD

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--out', default='/content/work')
a = p.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                         transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
test_ds = datasets.ImageFolder(f'{a.data}/test', tf)
print('class_to_idx:', test_ds.class_to_idx)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

model = torchvision.models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(f'{a.out}/models/baseline.pth', map_location=device))
model = model.to(device).eval()

probs, labels = [], []
with torch.no_grad():
    for x, y in test_dl:
        probs += torch.sigmoid(model(x.to(device))).squeeze(1).cpu().tolist()
        labels += y.tolist()
probs, labels = np.array(probs), np.array(labels)
preds = (probs >= 0.5).astype(int)

metrics = {
    'accuracy': accuracy_score(labels, preds),
    'precision': precision_score(labels, preds),
    'recall': recall_score(labels, preds),
    'f1': f1_score(labels, preds),
    'roc_auc': roc_auc_score(labels, probs),
}
print(metrics)
json.dump(metrics, open(f'{a.out}/results/metrics.json', 'w'), indent=2)

# confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap='Blues')
plt.xticks([0, 1], ['fake', 'real']); plt.yticks([0, 1], ['fake', 'real'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion matrix (test)')
plt.savefig(f'{a.out}/results/confusion_matrix.png', dpi=150, bbox_inches='tight'); plt.close()

# ROC
fpr, tpr, _ = roc_curve(labels, probs)
plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
plt.title(f"ROC (AUC = {metrics['roc_auc']:.3f})")
plt.savefig(f'{a.out}/results/roc_curve.png', dpi=150, bbox_inches='tight'); plt.close()

# training curves
h = json.load(open(f'{a.out}/results/history.json'))
ep = range(1, len(h['train_loss']) + 1)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(ep, h['train_loss'], label='train'); ax[0].plot(ep, h['val_loss'], label='valid')
ax[0].set_title('Loss'); ax[0].set_xlabel('Epoch'); ax[0].set_xticks(list(ep)); ax[0].legend()
ax[1].plot(ep, h['train_acc'], label='train'); ax[1].plot(ep, h['val_acc'], label='valid')
ax[1].set_title('Accuracy'); ax[1].set_xlabel('Epoch'); ax[1].set_xticks(list(ep)); ax[1].legend()
plt.savefig(f'{a.out}/results/training_curves.png', dpi=150, bbox_inches='tight'); plt.close()

# misclassified images: 8 fakes called real + 8 reals called fake
fake_as_real = np.where((labels == 0) & (preds == 1))[0][:8]
real_as_fake = np.where((labels == 1) & (preds == 0))[0][:8]
shown = list(fake_as_real) + list(real_as_fake)
fig, axes = plt.subplots(4, 4, figsize=(10, 11))
for axx in axes.ravel():
    axx.axis('off')
for axx, i in zip(axes.ravel(), shown):
    axx.imshow(Image.open(test_ds.samples[i][0]).convert('RGB'))
    true = 'real' if labels[i] == 1 else 'fake'
    axx.set_title(f'true: {true}\nP(real)={probs[i]:.2f}', fontsize=9)
fig.subplots_adjust(hspace=0.35)
plt.savefig(f'{a.out}/results/misclassified.png', dpi=130, bbox_inches='tight'); plt.close()
print('Saved all results.')
