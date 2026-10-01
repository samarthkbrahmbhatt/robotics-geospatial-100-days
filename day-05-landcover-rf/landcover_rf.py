"""
Day 05: Land cover classification on EuroSAT with a Random Forest.

EuroSAT is 27,000 Sentinel-2 image chips (64x64 px) labelled into 10
land cover classes. This script:

  1. downloads the RGB version if it is not already on disk
  2. turns each image into a short feature vector by hand
  3. trains a Random Forest and tests it on data it has never seen
  4. reports accuracy, a confusion matrix and feature importances

The point is the evaluation, not the score. Hand-built features on RGB
will not beat a CNN, and the confusion matrix shows exactly where they
fall down.
"""

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------- Settings ----------------
DATA_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
DATA_DIR = Path(__file__).parent / "data"
MAX_PER_CLASS = 800      # keep the run to a couple of minutes; raise for a better model
TEST_FRACTION = 0.2
SEED = 42

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


def find_class_root(root: Path):
    """Locate the folder that directly contains the 10 class folders."""
    if not root.exists():
        return None
    for candidate in [root] + [p for p in root.iterdir() if p.is_dir()]:
        if sum((candidate / c).is_dir() for c in CLASSES) >= 8:
            return candidate
    return None


def download_dataset():
    """Fetch and unzip EuroSAT RGB (about 95 MB) if it is not already here."""
    existing = find_class_root(DATA_DIR)
    if existing:
        print(f"Dataset already present at {existing}")
        return existing

    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading EuroSAT RGB (about 95 MB). This happens once.")
    try:
        with urlopen(DATA_URL, timeout=120) as response:
            payload = response.read()
    except Exception as e:
        raise SystemExit(
            f"\nDownload failed: {e}\n"
            f"Download it manually from https://zenodo.org/records/7711810 "
            f"(the EuroSAT_RGB.zip file), then unzip it into:\n  {DATA_DIR}\n"
        )

    print(f"Downloaded {len(payload) / 1e6:.0f} MB. Extracting...")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(DATA_DIR)

    root = find_class_root(DATA_DIR)
    if root is None:
        raise SystemExit(f"Extracted, but no class folders found under {DATA_DIR}")
    return root


def extract_features(path):
    """Turn one 64x64 RGB chip into a handful of numbers.

    No deep learning here: just colour statistics and a crude texture
    measure. The feature importance plot shows which of these the
    forest actually relies on.
    """
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    features = []
    for band in (r, g, b):
        features += [band.mean(), band.std(),
                     np.percentile(band, 10), np.percentile(band, 90)]

    # Colour ratios: crude stand-ins for vegetation and water indices,
    # which normally need the near-infrared band EuroSAT RGB does not have.
    eps = 1e-6
    features.append(float(((g - r) / (g + r + eps)).mean()))   # greenness
    features.append(float(((b - r) / (b + r + eps)).mean()))   # blueness
    features.append(float(img.mean()))                          # overall brightness

    # Texture: how much neighbouring pixels differ. Cities are rough,
    # water is smooth.
    grey = img.mean(axis=2)
    features.append(float(np.abs(np.diff(grey, axis=0)).mean()))
    features.append(float(np.abs(np.diff(grey, axis=1)).mean()))

    return features


FEATURE_NAMES = ([f"{c}_{s}" for c in "RGB" for s in ("mean", "std", "p10", "p90")]
                 + ["greenness", "blueness", "brightness", "texture_v", "texture_h"])


def load_dataset(root):
    X, y = [], []
    rng = np.random.default_rng(SEED)

    for label in CLASSES:
        files = sorted((root / label).glob("*.jpg"))
        if not files:
            print(f"  WARNING: no images found for {label}")
            continue
        if len(files) > MAX_PER_CLASS:
            idx = rng.choice(len(files), MAX_PER_CLASS, replace=False)
            files = [files[i] for i in idx]
        for f in files:
            X.append(extract_features(f))
            y.append(label)
        print(f"  {label:<22} {len(files)} images")

    return np.array(X, dtype=np.float32), np.array(y)


def plot_results(y_test, y_pred, forest):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES, normalize="true")
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] > 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- Feature importance ---
    order = np.argsort(forest.feature_importances_)
    ax = axes[1]
    ax.barh([FEATURE_NAMES[i] for i in order],
            forest.feature_importances_[order], color="tab:green")
    ax.set_title("What the forest actually uses")
    ax.set_xlabel("Importance")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Day 05: EuroSAT land cover with a Random Forest", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "landcover_rf.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


def main():
    root = download_dataset()

    print(f"\nLoading up to {MAX_PER_CLASS} images per class...")
    X, y = load_dataset(root)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features each")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y)
    print(f"Train: {len(X_train)}   Test: {len(X_test)} (never seen during training)")

    # Baselines worth beating
    chance = 1 / len(set(y))
    tree = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    tree_acc = accuracy_score(y_test, tree.predict(X_test))

    forest = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nRandom guessing:   {100 * chance:.1f}%")
    print(f"Single tree:       {100 * tree_acc:.1f}%")
    print(f"Random forest:     {100 * acc:.1f}%")

    print("\nPer class:")
    print(classification_report(y_test, y_pred, digits=2))

    # Which pairs does it mix up most?
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES, normalize="true")
    np.fill_diagonal(cm, 0)
    pairs = [(cm[i, j], i, j)
             for i in range(len(CLASSES)) for j in range(len(CLASSES))
             if i != j and cm[i, j] > 0]
    if pairs:
        print("Most confused pairs:")
        for rate, i, j in sorted(pairs, reverse=True)[:3]:
            print(f"  {CLASSES[i]} predicted as {CLASSES[j]}: {100 * rate:.0f}% of the time")
    else:
        print("No confusions at all, which usually means the task is too easy "
              "or something is leaking between train and test.")

    plot_results(y_test, y_pred, forest)


if __name__ == "__main__":
    main()
