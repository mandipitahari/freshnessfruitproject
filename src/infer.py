"""
Modular inference with a 2-logit head and metadata-driven label mapping.

Public API:
- load_metadata() -> dict
- load_model(model_path=..., meta=...) -> nn.Module
- preprocess_image(path) -> torch.Tensor[1,3,224,224]
- predict_one(model, img_path, meta, return_probs=False) -> (label, prob[label])
- predict_dir(model, dir_path, meta, csv_path=None) -> list[(path, label, prob)]
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import argparse, csv, json

import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "apple_effb0_ce.pt"
META_PATH    = PROJECT_ROOT / "models" / "metadata.json"

# Your example defaults
DEFAULT_IMG = Path("/home/mandip/PycharmProjects/FreshnessFruitProject/data/processed/apple_only/test/rotten/rotated_by_15_Screen Shot 2018-06-07 at 2.20.04 PM.jpg")
DEFAULT_DIR = Path("/home/mandip/PycharmProjects/FreshnessFruitProject/data/processed/apple_only/val")

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Same normalization as training
_PREP = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------- Utils ----------
def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_metadata(meta_path: Path = META_PATH) -> Dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}. Train first to create it.")
    return json.loads(meta_path.read_text())

def preprocess_image(img_path: Path) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    x = _PREP(img).unsqueeze(0)  # [1,3,224,224]
    return x

# ---------- Model ----------
def load_model(model_path: Path = MODEL_PATH, meta: Optional[Dict] = None) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)  # 2-class head
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model

# ---------- Predictors ----------
@torch.no_grad()
def predict_one(model: nn.Module,
                img_path: Path,
                meta: Dict,
                return_probs: bool = False) -> Tuple[str, float] | Tuple[str, Dict[str, float]]:
    """
    Predict one image using softmax and metadata mapping.
    Returns (label, top_prob) or (label, {'fresh': p0, 'rotten': p1})
    """
    if not img_path.exists() or not is_image(img_path):
        raise FileNotFoundError(f"Image not found or unsupported: {img_path}")

    x = preprocess_image(img_path).to(DEVICE)
    logits = model(x)                        # [1,2]
    probs  = torch.softmax(logits, dim=1)[0] # [2]

    # Map indices back to labels using metadata
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    top_idx = int(torch.argmax(probs).item())
    top_label = idx_to_class[top_idx]
    top_prob  = float(probs[top_idx].item())

    if return_probs:
        by_label = {idx_to_class[i]: float(probs[i].item()) for i in range(len(probs))}
        return top_label, by_label
    return top_label, top_prob

@torch.no_grad()
def predict_dir(model: nn.Module,
                dir_path: Path,
                meta: Dict,
                csv_path: Optional[Path] = None) -> List[Tuple[Path, str, float]]:
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    images = sorted([p for p in dir_path.rglob("*") if p.is_file() and is_image(p)])
    results: List[Tuple[Path, str, float]] = []
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}

    for p in images:
        label, probs = predict_one(model, p, meta, return_probs=True)
        print(f"{p.name}: {label} (fresh={probs.get('fresh',0):.4f}, rotten={probs.get('rotten',0):.4f})")
        results.append((p, label, float(probs[label])))

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "probability"])
            for p, lbl, pr in results:
                w.writerow([str(p), lbl, f"{pr:.6f}"])
        print(f"Saved CSV: {csv_path}")

    print(f"\nProcessed {len(results)} image(s).")
    return results

# ---------- CLI ----------
def _parse_args():
    parser = argparse.ArgumentParser(description="Apple freshness inference (2-class softmax)")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--img", type=str,
                   help=f"Path to a single image (default sample): {DEFAULT_IMG}")
    g.add_argument("--dir", type=str,
                   help=f"Path to a directory of images (default: {DEFAULT_DIR})")
    parser.add_argument("--csv", type=str, default="",
                        help="Optional CSV path for directory mode (e.g., outputs/preds.csv)")
    args = parser.parse_args()
    if not args.img and not args.dir:
        args.dir = str(DEFAULT_DIR)
    return args

def main():
    args = _parse_args()
    meta = load_metadata(META_PATH)
    print("Device:", DEVICE)
    print("Loading model:", MODEL_PATH)
    print("Metadata:", META_PATH, "| classes:", meta["class_to_idx"])

    model = load_model(MODEL_PATH, meta)
    if args.img:
        label, probs = predict_one(model, Path(args.img), meta, return_probs=True)
        print(f"{Path(args.img).name}: {label} {probs}")
    else:
        csv_path = Path(args.csv) if args.csv else None
        predict_dir(model, Path(args.dir), meta, csv_path)


if __name__ == "__main__":
    model = load_model()
    meta  = load_metadata()
    label, prob = predict_one(
        model,
        Path("/home/mandip/PycharmProjects/FreshnessFruitProject/data/processed/apple_only/test/fresh/rotated_by_30_Screen Shot 2018-06-08 at 5.06.47 PM.jpg"),
        meta
    )
    print(label, prob)
