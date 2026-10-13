"""
Train an EfficientNet-B0 CNN model to classify apple freshness (fresh vs rotten).

Changes added:
- Logs training history (acc/loss) to models/history.csv
- Saves test predictions with probabilities to models/test_preds.csv
- Computes and saves summary metrics to models/metrics_summary.csv
- Saves confusion matrix numbers to models/confusion_matrix.csv
- Keeps your original structure and paths

Run:
    python train.py
"""


from pathlib import Path
import csv
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)

# ===============================
# CONFIGURATION
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT   = PROJECT_ROOT / "data" / "processed" / "apple_only"
MODEL_DIR   = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # create models/ if missing

# Training parameters
IMG_SIZE = 224          # images already resized to 224x224
BATCH = 32              # batch size
EPOCHS = 10             # number of training epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ===============================
# DATA PREPARATION
# ===============================
tfm_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
tfm_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Datasets
train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=tfm_train)
val_ds   = datasets.ImageFolder(DATA_ROOT / "val",   transform=tfm_eval)
test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=tfm_eval)

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ===============================
# MODEL SETUP (EfficientNet-B0 CNN)
# ===============================
weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = torchvision.models.efficientnet_b0(weights=weights)

# Freeze backbone
for p in model.parameters():
    p.requires_grad = False

# Replace classifier head for binary classification (fresh vs rotten)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)

model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ===============================
# TRAINING UTIL
# ===============================
def run_epoch(loader, train=False):
    """Run one epoch: forward pass, loss calculation, backprop if training."""
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.float().to(DEVICE)

        if train:
            optimizer.zero_grad()

        logits = model(x).squeeze(1)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.numel()

    return correct/total, loss_sum/total

# ===============================
# TRAINING LOOP + LOGGING
# ===============================
history = {"epoch": [], "train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

best_val = 0.0
for epoch in range(1, EPOCHS+1):
    tr_acc, tr_loss = run_epoch(train_loader, train=True)
    va_acc, va_loss = run_epoch(val_loader, train=False)

    print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f} | val acc {va_acc:.3f} loss {va_loss:.3f}")

    history["epoch"].append(epoch)
    history["train_acc"].append(tr_acc)
    history["train_loss"].append(tr_loss)
    history["val_acc"].append(va_acc)
    history["val_loss"].append(va_loss)

    # Save best model
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), MODEL_DIR / "apple_effb0.pt")

# save history
with open(MODEL_DIR / "history.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_acc", "train_loss", "val_acc", "val_loss"])
    for i in range(len(history["epoch"])):
        w.writerow([history["epoch"][i], history["train_acc"][i], history["train_loss"][i],
                    history["val_acc"][i], history["val_loss"][i]])

print("✅ Training complete. Best val acc:", best_val)

# ===============================
# FINAL EVALUATION
# ===============================
model.load_state_dict(torch.load(MODEL_DIR / "apple_effb0.pt", map_location=DEVICE))
model.eval()

y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)

        y_prob.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().astype(int).tolist())

# Metrics
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
auc = roc_auc_score(y_true, y_prob)

print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["rotten","fresh"]))
print(f"\nAccuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

# Save per-image predictions
with open(MODEL_DIR / "test_preds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["y_true", "y_pred", "y_prob"])
    for t, p, s in zip(y_true, y_pred, y_prob):
        w.writerow([t, p, f"{s:.6f}"])

# Save metrics summary
with open(MODEL_DIR / "metrics_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Split", "Accuracy", "Precision", "Recall", "F1", "AUC"])
    w.writerow(["EfficientNet-B0 (frozen backbone)", "Test",
                f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{auc:.4f}"])

# Save confusion matrix numbers
cm = confusion_matrix(y_true, y_pred)
with open(MODEL_DIR / "confusion_matrix.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["", "Pred:0(rotten)", "Pred:1(fresh)"])
    w.writerow(["True:0(rotten)", cm[0,0], cm[0,1]])
    w.writerow(["True:1(fresh)", cm[1,0], cm[1,1]])

print("✅ Saved: history.csv, test_preds.csv, metrics_summary.csv, confusion_matrix.csv in", MODEL_DIR)