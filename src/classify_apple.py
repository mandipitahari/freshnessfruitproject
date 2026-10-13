"""
Resume-safe CLI tool for apple freshness classification.
Skips images already copied to fresh_app or rotten_app.
"""

from pathlib import Path
import shutil
import torch, torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
import json
import argparse

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
APPLE_DIR = DATA_DIR / "apple"
FRESH_DIR = DATA_DIR / "fresh_app"
ROTTEN_DIR = DATA_DIR / "rotten_app"

MODEL_PATH = PROJECT_ROOT / "models" / "apple_effb0_ce.pt"
META_PATH = PROJECT_ROOT / "models" / "metadata.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

FRESH_DIR.mkdir(parents=True, exist_ok=True)
ROTTEN_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load metadata
# -------------------------
def load_metadata():
    return json.loads(META_PATH.read_text())

META = load_metadata()
IDX_TO_CLASS = {int(k): v for k, v in META["idx_to_class"].items()}

# -------------------------
# Preprocessing
# -------------------------
PREP = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# -------------------------
# Model
# -------------------------
def load_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval().to(DEVICE)
    return model

model = load_model()

# -------------------------
# Prediction
# -------------------------
@torch.no_grad()
def predict_image(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    x = PREP(img).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)[0]
    idx = torch.argmax(probs).item()
    return IDX_TO_CLASS[idx], float(probs[idx])

# -------------------------
# Resume Logic
# -------------------------
def already_processed(filename: str) -> bool:
    return (
        (FRESH_DIR / filename).exists()
        or (ROTTEN_DIR / filename).exists()
    )

def process_images(threshold: float):
    images = list(APPLE_DIR.rglob("*.jpg")) + list(APPLE_DIR.rglob("*.png"))

    total = len(images)
    skipped = 0
    processed = 0

    print(f"📂 Total images to check: {total}")

    for img_path in images:
        if already_processed(img_path.name):
            skipped += 1
            continue

        label, confidence = predict_image(img_path)

        if confidence < threshold:
            print(f"⚠️ Low confidence skipped: {img_path.name}")
            continue

        target_dir = FRESH_DIR if "fresh" in label.lower() else ROTTEN_DIR
        shutil.copy2(img_path, target_dir / img_path.name)

        processed += 1
        print(f"✅ {img_path.name} → {label} ({confidence:.2f})")

    print("\n📊 Summary")
    print(f"Already processed: {skipped}")
    print(f"Newly processed:   {processed}")
    print("🎉 Resume complete!")

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    args = parser.parse_args()

    process_images(args.threshold)
