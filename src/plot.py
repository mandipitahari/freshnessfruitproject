"""
plot.py — Build all figures and a single PDF report from outputs saved by train.py

Inputs (from models/):
- history.csv
- test_preds.csv
- metrics_summary.csv
- confusion_matrix.csv

Outputs:
- figures: fig_acc_curve.png, fig_loss_curve.png, fig_roc_curve.png, fig_confusion_matrix.png
- pdf: results_report.pdf (contains all figures + a metrics table page)

Run:
    python plot.py
"""


from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc, confusion_matrix


# IMPORTANT: When making charts, use matplotlib, single plot per figure, no specific colors.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Load CSVs
# -------------------
history_path = MODEL_DIR / "history.csv"
metrics_path = MODEL_DIR / "metrics_summary.csv"
preds_path   = MODEL_DIR / "test_preds.csv"

# History
epochs, tr_acc, va_acc, tr_loss, va_loss = [], [], [], [], []
with open(history_path, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        epochs.append(int(row["epoch"]))
        tr_acc.append(float(row["train_acc"]))
        va_acc.append(float(row["val_acc"]))
        tr_loss.append(float(row["train_loss"]))
        va_loss.append(float(row["val_loss"]))

# Metrics table
metrics_df = pd.read_csv(metrics_path)

# Predictions
preds_df = pd.read_csv(preds_path)
y_true = preds_df["y_true"].values.astype(int)
y_pred = preds_df["y_pred"].values.astype(int)
y_prob = preds_df["y_prob"].values.astype(float)

# -------------------
# Build Figures (PNG)
# -------------------

# 1) Accuracy curve
plt.figure()
plt.plot(epochs, tr_acc, label="Train")
plt.plot(epochs, va_acc, label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
acc_png = MODEL_DIR / "fig_acc_curve.png"
plt.savefig(acc_png, dpi=300)
plt.close()

# 2) Loss curve
plt.figure()
plt.plot(epochs, tr_loss, label="Train")
plt.plot(epochs, va_loss, label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_png = MODEL_DIR / "fig_loss_curve.png"
plt.savefig(loss_png, dpi=300)
plt.close()

# 3) ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_png = MODEL_DIR / "fig_roc_curve.png"
plt.savefig(roc_png, dpi=300)
plt.close()

# 4) Confusion matrix (from y_true/y_pred to be safe)
cm = confusion_matrix(y_true, y_pred)
labels = ["rotten", "fresh"]
plt.figure()
im = plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Test Set)")
plt.colorbar(im, fraction=0.046, pad=0.04)
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45, ha="right")
plt.yticks(tick_marks, labels)
th = cm.max() / 2.0 if cm.max() > 0 else 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > th else "black")
plt.ylabel("True label"); plt.xlabel("Predicted label")
plt.tight_layout()
cm_png = MODEL_DIR / "fig_confusion_matrix.png"
plt.savefig(cm_png, dpi=300)
plt.close()

# -------------------
# Build PDF
# -------------------
pdf_path = MODEL_DIR / "results_report.pdf"
with PdfPages(pdf_path) as pdf:
    # Cover page with key metrics
    fig = plt.figure()
    fig.suptitle("Apple Freshness Classification — Results Summary", fontsize=14)
    plt.axis("off")
    # Show metrics table as text
    txt_lines = ["Summary Metrics (Test Set):"]
    for _, row in metrics_df.iterrows():
        txt_lines.append(
            f"{row['Model']} | {row['Split']} | "
            f"Acc={row['Accuracy']}  Prec={row['Precision']}  "
            f"Rec={row['Recall']}  F1={row['F1']}  AUC={row['AUC']}"
        )
    plt.text(0.02, 0.9, "\n".join(txt_lines), va="top", fontfamily="monospace")
    pdf.savefig(fig); plt.close(fig)

    # Accuracy curve
    fig = plt.figure()
    img = plt.imread(str(acc_png))
    plt.imshow(img); plt.axis("off")
    pdf.savefig(fig); plt.close(fig)

    # Loss curve
    fig = plt.figure()
    img = plt.imread(str(loss_png))
    plt.imshow(img); plt.axis("off")
    pdf.savefig(fig); plt.close(fig)

    # ROC curve
    fig = plt.figure()
    img = plt.imread(str(roc_png))
    plt.imshow(img); plt.axis("off")
    pdf.savefig(fig); plt.close(fig)

    # Confusion matrix
    fig = plt.figure()
    img = plt.imread(str(cm_png))
    plt.imshow(img); plt.axis("off")
    pdf.savefig(fig); plt.close(fig)

    # Metrics table page rendered via matplotlib table
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_title('Comparison Table', pad=10)
    table = ax.table(cellText=metrics_df.values,
                     colLabels=metrics_df.columns,
                     loc='center')
    table.scale(1, 1.4)
    pdf.savefig(fig); plt.close(fig)

print(f"✅ Saved figures and PDF to: {pdf_path}")