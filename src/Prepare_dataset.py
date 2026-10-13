from pathlib import Path
import random, shutil
from PIL import Image

random.seed(42)

# 1. Source = raw data
SRC = Path(__file__).resolve().parent.parent / "data" / "raw"
FRESH = SRC / "fresh_apple"
STALE = SRC / "stale_apple"

# 2. Output = processed/apple_only (overwrite existing!)
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "apple_only"
if OUT.exists():
    shutil.rmtree(OUT)   # ⚠️ deletes previous version
for split in ["train", "val", "test"]:
    for cls in ["fresh", "rotten"]:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)

# 3. Collect images
def list_imgs(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".png"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

fresh_imgs = list_imgs(FRESH)
rotten_imgs = list_imgs(STALE)

# 4. Resize + split 70/15/15
def resize_and_save(src, dst, size=(224,224)):
    img = Image.open(src).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    img.save(dst, format="JPEG", quality=90, optimize=True)

def split_and_copy(imgs, label):
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    train, val, test = imgs[:n_train], imgs[n_train:n_train+n_val], imgs[n_train+n_val:]
    for group, split in zip([train, val, test], ["train","val","test"]):
        for src in group:
            dst = OUT / split / label / (src.stem + ".jpg")  # force .jpg
            resize_and_save(src, dst)
    return len(train), len(val), len(test)

t_fresh  = split_and_copy(fresh_imgs, "fresh")
t_rotten = split_and_copy(rotten_imgs, "rotten")

print("✅ Apple-only dataset (compressed 224×224) created at:", OUT.resolve())
print("Fresh counts  (train, val, test):", t_fresh)
print("Rotten counts (train, val, test):", t_rotten)
