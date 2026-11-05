import argparse, json, random
import numpy as np, torch, torch.nn as nn, torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import IMG_SIZE, MEAN, STD, SEED

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--out', default='/content/work')
p.add_argument('--epochs', type=int, default=4)
p.add_argument('--bs', type=int, default=64)
p.add_argument('--lr', type=float, default=1e-4)
a = p.parse_args()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
train_ds = datasets.ImageFolder(f'{a.data}/train', tf)
valid_ds = datasets.ImageFolder(f'{a.data}/valid', tf)
print('class_to_idx:', train_ds.class_to_idx)
assert train_ds.class_to_idx == {'fake': 0, 'real': 1}, 'Class mapping is wrong!'

train_dl = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=2)
valid_dl = DataLoader(valid_ds, batch_size=a.bs, shuffle=False, num_workers=2)

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)
crit = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=a.lr)

def run_epoch(dl, train):
    model.train(train)
    loss_sum = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += ((out > 0).float() == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best = 0.0
for ep in range(a.epochs):
    tl, ta = run_epoch(train_dl, True)
    vl, va = run_epoch(valid_dl, False)
    for k, v in zip(history, [tl, ta, vl, va]):
        history[k].append(v)
    print(f'epoch {ep+1}/{a.epochs} | train acc {ta:.3f} | val acc {va:.3f}')
    if va > best:
        best = va
        torch.save(model.state_dict(), f'{a.out}/models/baseline.pth')
        print('  -> saved baseline.pth')

json.dump(history, open(f'{a.out}/results/history.json', 'w'))
print('Done. Best val acc:', round(best, 4))
