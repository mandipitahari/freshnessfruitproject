"""
Gradio web app for apple freshness inference.
"""

from pathlib import Path
import torch, torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import json

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "apple_effb0_ce.pt"
META_PATH = PROJECT_ROOT / "models" / "metadata.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224


# -------------------------
# Load metadata
# -------------------------
def load_metadata(meta_path: Path = META_PATH):
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    return json.loads(meta_path.read_text())


META = load_metadata()

# -------------------------
# Preprocessing
# -------------------------
PREP = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# -------------------------
# Model
# -------------------------
def load_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)  # two logits
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model


model = load_model()


# -------------------------
# Prediction
# -------------------------
@torch.no_grad()
def predict(img: Image.Image):
    x = PREP(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(x)                 # [1,2]
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx_to_class = {int(k): v for k, v in META["idx_to_class"].items()}
    results = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
    return results  # ✅ only return the dict

# -------------------------
# Gradio UI
# -------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Apple Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🍎 Apple Freshness Classifier",
    description="Upload an apple image to check if it's fresh or rotten"
)

if __name__ == "__main__":
    demo.launch()
