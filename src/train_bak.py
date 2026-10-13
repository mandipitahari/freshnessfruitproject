"""
Train an EfficientNet-B0 CNN model to classify apple freshness (fresh vs rotten).
Dataset: data/processed/apple_only/ (images already resized to 224x224).
Output: Trained model weights stored in models/apple_effb0.pt
"""

from pathlib import Path
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# CONFIGURATION
# ===============================
# Paths
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
# Since images are already 224x224, no resize needed
# Apply only normalization and some augmentation for training
tfm_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),                       # random flip for augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),    # random brightness/contrast
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),            # standard ImageNet normalization
])
tfm_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load datasets
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
# Load EfficientNet-B0 pretrained on ImageNet
weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = torchvision.models.efficientnet_b0(weights=weights)

# Freeze backbone parameters (feature extractor)
for p in model.parameters():
    p.requires_grad = False

# Replace classifier head for binary classification (fresh vs rotten)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)  # output single logit

model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()               # binary cross-entropy loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ===============================
# TRAINING FUNCTION
# ===============================
def run_epoch(loader, train=False):
    """Run one epoch: forward pass, loss calculation, backprop if training."""
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.float().to(DEVICE)   # move to GPU/CPU

        if train:
            optimizer.zero_grad()

        logits = model(x).squeeze(1)                # forward pass
        loss = criterion(logits, y)                 # loss

        if train:
            loss.backward()
            optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.numel()

    return correct/total, loss_sum/total

# ===============================
# TRAINING LOOP
# ===============================
best_val = 0
for epoch in range(1, EPOCHS+1):
    tr_acc, tr_loss = run_epoch(train_loader, train=True)
    va_acc, va_loss = run_epoch(val_loader, train=False)
    print(f"Epoch {epoch:02d} | "
          f"train acc {tr_acc:.3f} loss {tr_loss:.3f} | "
          f"val acc {va_acc:.3f} loss {va_loss:.3f}")

    # Save best model so far
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), MODEL_DIR / "apple_effb0.pt")

print("✅ Training complete. Best val acc:", best_val)

# ===============================
# FINAL EVALUATION
# ===============================
# Load best model
model.load_state_dict(torch.load(MODEL_DIR / "apple_effb0.pt", map_location=DEVICE))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        p = (torch.sigmoid(model(x)).cpu().numpy().ravel() >= 0.5).astype(int)
        y_pred.extend(p.tolist())
        y_true.extend(y.numpy().astype(int).tolist())

print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["rotten","fresh"]))
