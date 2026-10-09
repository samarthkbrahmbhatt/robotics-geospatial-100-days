"""
Day 13: Was Day 05's 80.9% real, or a lucky split?

Day 05 trained a Random Forest on EuroSAT, measured 80.9% on one held-out
test set, and stopped there. That number came from a single random split,
and a single split is one sample from a distribution. It carries no
indication of how much it would move if the split had fallen differently.

This answers four questions the single split cannot:

  1. How much does accuracy vary across folds? (cross-validation)
  2. Does more training data help, or are the features the ceiling?
     (learning curve)
  3. Does tuning the forest help? (hyperparameter search)
  4. Is any tuning gain larger than the fold-to-fold noise, or is it
     inside the margin of error?

Question 4 is the one that matters. A tuning gain smaller than the spread
between folds is not a result.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, learning_curve)

# ---------------- Settings ----------------
MAX_PER_CLASS = 800          # same as Day 05, so the numbers are comparable
N_FOLDS = 5
SEED = 42

HERE = Path(__file__).parent
CACHE = HERE / "features_cache.npz"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

FEATURE_NAMES = ([f"{c}_{s}" for c in "RGB" for s in ("mean", "std", "p10", "p90")]
                 + ["greenness", "blueness", "brightness", "texture_v", "texture_h"])


def find_dataset():
    """Reuse the EuroSAT download from Day 05 rather than fetching it again."""
    candidates = [HERE.parent / "day-05-landcover-rf" / "data", HERE / "data"]
    for root in candidates:
        if not root.exists():
            continue
        for folder in [root] + [p for p in root.iterdir() if p.is_dir()]:
            if sum((folder / c).is_dir() for c in CLASSES) >= 8:
                return folder
    raise SystemExit(
        "Could not find the EuroSAT images.\n"
        "They should already be in day-05-landcover-rf/data from Day 05.\n"
        "If that folder is gone, rerun Day 05's script once to fetch them."
    )


def extract_features(path):
    """Identical to Day 05, so the comparison is like for like."""
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    features = []
    for band in (r, g, b):
        features += [band.mean(), band.std(),
                     np.percentile(band, 10), np.percentile(band, 90)]

    eps = 1e-6
    features.append(float(((g - r) / (g + r + eps)).mean()))
    features.append(float(((b - r) / (b + r + eps)).mean()))
    features.append(float(img.mean()))

    grey = img.mean(axis=2)
    features.append(float(np.abs(np.diff(grey, axis=0)).mean()))
    features.append(float(np.abs(np.diff(grey, axis=1)).mean()))
    return features


def load_features():
    """Extract features once and cache them.

    Reading 8000 JPEGs takes a couple of minutes and the result never
    changes, so caching makes every later experiment on this data cheap.
    """
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)
        print(f"Loaded cached features: {cached['X'].shape[0]} samples")
        return cached["X"], cached["y"]

    root = find_dataset()
    print(f"Extracting features from {root} (cached after this run)...")

    rng = np.random.default_rng(SEED)
    X, y = [], []
    for label in CLASSES:
        files = sorted((root / label).glob("*.jpg"))
        if len(files) > MAX_PER_CLASS:
            idx = rng.choice(len(files), MAX_PER_CLASS, replace=False)
            files = [files[i] for i in idx]
        for f in files:
            X.append(extract_features(f))
            y.append(label)
        print(f"  {label:<22} {len(files)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    np.savez_compressed(CACHE, X=X, y=y)
    print(f"Cached to {CACHE.name}")
    return X, y


def self_test():
    """Check that the evaluation would actually detect a broken model.

    A cross-validation harness that reports a good score on pure noise is
    broken, and that failure is silent. So the harness is pointed at
    random labels first: it has to come back at chance.
    """
    rng = np.random.default_rng(0)
    X_noise = rng.normal(size=(600, 17))
    y_noise = rng.choice(CLASSES, size=600)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=40, random_state=SEED, n_jobs=-1),
        X_noise, y_noise, cv=cv)
    chance = 1 / len(CLASSES)
    print(f"  random features, random labels -> {scores.mean():.3f} "
          f"(chance is {chance:.3f})")
    assert scores.mean() < 0.2, (
        f"the harness scored {scores.mean():.3f} on pure noise, so it is leaking")

    # And a perfectly separable problem has to come back near 1.0
    y_easy = np.array([CLASSES[i % 10] for i in range(600)])
    X_easy = np.zeros((600, 17), dtype=np.float32)
    for i, label in enumerate(y_easy):
        X_easy[i, CLASSES.index(label)] = 1.0
    easy = cross_val_score(
        RandomForestClassifier(n_estimators=40, random_state=SEED, n_jobs=-1),
        X_easy, y_easy, cv=cv)
    print(f"  perfectly separable classes    -> {easy.mean():.3f} (expect 1.000)")
    assert easy.mean() > 0.99, "the harness failed a problem it should ace"

    # The learning curve must not be handed subsets that are missing classes.
    # Features are loaded class by class, so a prefix of a training fold is
    # a prefix of the class list unless the subsets are shuffled first.
    y_sorted = np.repeat(CLASSES, 60)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    train_idx, _ = next(folds.split(np.zeros(len(y_sorted)), y_sorted))
    prefix_classes = len(set(y_sorted[train_idx[:len(train_idx) // 10]]))
    shuffled = rng.permutation(train_idx)
    shuffled_classes = len(set(y_sorted[shuffled[:len(shuffled) // 10]]))
    print(f"  smallest learning-curve subset covers {prefix_classes}/10 classes "
          f"unshuffled, {shuffled_classes}/10 shuffled")
    assert shuffled_classes > prefix_classes, (
        "shuffling should widen the class coverage of a small subset")


def main():
    print("Self-test of the evaluation harness:")
    self_test()

    X, y = load_features()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(set(y))} classes")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    baseline = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)

    # ---------- 1. How much does the score move between folds? ----------
    print(f"\n--- {N_FOLDS}-fold cross-validation, Day 05's exact model ---")
    scores = cross_val_score(baseline, X, y, cv=cv, n_jobs=1)
    for i, s in enumerate(scores, 1):
        print(f"  fold {i}: {100 * s:.2f}%")
    print(f"  mean {100 * scores.mean():.2f}%  std {100 * scores.std():.2f}%  "
          f"spread {100 * (scores.max() - scores.min()):.2f} points")
    print(f"  Day 05 reported 80.90% from a single split.")

    # ---------- 2. Does more data help? ----------
    print("\n--- Learning curve ---")
    # shuffle=True is essential here. learning_curve takes a PREFIX of each
    # training fold, and the features are loaded class by class, so without
    # shuffling the smallest subset contains one class out of ten. The curve
    # then looks like a dramatic improvement with more data when it is really
    # just classes being added back in.
    sizes, train_scores, val_scores = learning_curve(
        baseline, X, y, cv=cv, n_jobs=1, shuffle=True, random_state=SEED,
        train_sizes=np.array([0.1, 0.25, 0.5, 0.75, 1.0]))
    for n, tr, va in zip(sizes, train_scores, val_scores):
        print(f"  {n:5d} training samples: train {100 * tr.mean():5.2f}%  "
              f"validation {100 * va.mean():5.2f}%  "
              f"gap {100 * (tr.mean() - va.mean()):5.2f} points")

    last_gain = 100 * (val_scores[-1].mean() - val_scores[-2].mean())
    print(f"  Going from {sizes[-2]} to {sizes[-1]} samples gained "
          f"{last_gain:+.2f} points.")

    # ---------- 3. Does tuning help? ----------
    print("\n--- Hyperparameter search ---")
    # n_estimators is deliberately NOT in the grid. Adding trees to a random
    # forest reduces variance and never causes overfitting, so it trades
    # compute for a little stability rather than being a real accuracy knob.
    # Searching over it mostly wastes time. The three parameters here all
    # control how much each individual tree is allowed to memorise, which
    # is where overfitting actually lives.
    grid = {
        "max_depth": [None, 12, 20],
        "max_features": ["sqrt", 0.5],
        "min_samples_leaf": [1, 3],
    }
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"  {n_combos} combinations x {N_FOLDS} folds at 300 trees each.")
    print(f"  This is the slow part, roughly {n_combos * 15 // 60} to "
          f"{n_combos * 30 // 60} minutes.")

    search = GridSearchCV(
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        grid, cv=cv, n_jobs=1, return_train_score=False)
    search.fit(X, y)

    print(f"  Best: {search.best_params_}")
    print(f"  Best cross-validated accuracy: {100 * search.best_score_:.2f}%")

    # ---------- 4. Is the gain real? ----------
    tuned_gain = 100 * (search.best_score_ - scores.mean())
    fold_std = 100 * scores.std()

    print("\n--- Is the tuning gain bigger than the noise? ---")
    print(f"  Untuned cross-validated accuracy : {100 * scores.mean():.2f}%")
    print(f"  Tuned cross-validated accuracy   : {100 * search.best_score_:.2f}%")
    print(f"  Gain from tuning                 : {tuned_gain:+.2f} points")
    print(f"  Fold-to-fold standard deviation  : {fold_std:.2f} points")

    if tuned_gain < fold_std:
        print("  The gain is SMALLER than the fold-to-fold spread, so tuning did "
              "not\n  reliably improve anything. The features are the limit, not "
              "the forest.")
    else:
        print("  The gain is larger than the fold-to-fold spread, so it is "
              "probably real.")

    results = {
        "cv_scores": [float(s) for s in scores],
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "day05_single_split": 0.809,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_score": float(search.best_score_),
        "tuning_gain_points": float(tuned_gain),
        "learning_curve_sizes": [int(n) for n in sizes],
        "learning_curve_val": [float(v.mean()) for v in val_scores],
        "learning_curve_train": [float(t.mean()) for t in train_scores],
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    ax.bar(range(1, N_FOLDS + 1), 100 * scores, color="tab:blue", alpha=0.85)
    ax.axhline(100 * scores.mean(), color="k", linestyle="--",
               label=f"CV mean {100 * scores.mean():.2f}%")
    ax.axhline(80.9, color="tab:red", linestyle=":",
               label="Day 05 single split 80.90%")
    ax.set_ylim(min(100 * scores.min(), 78) - 2, max(100 * scores.max(), 82) + 2)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("One number, or a distribution?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.plot(sizes, 100 * train_scores.mean(axis=1), "o-",
            color="tab:orange", label="Training")
    ax.fill_between(sizes, 100 * (val_scores.mean(axis=1) - val_scores.std(axis=1)),
                    100 * (val_scores.mean(axis=1) + val_scores.std(axis=1)),
                    color="tab:blue", alpha=0.15)
    ax.plot(sizes, 100 * val_scores.mean(axis=1), "o-",
            color="tab:blue", label="Validation")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Would more data help?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = ["Day 05\nsingle split", f"{N_FOLDS}-fold CV\nuntuned", "Tuned"]
    values = [80.9, 100 * scores.mean(), 100 * search.best_score_]
    errors = [0, fold_std, fold_std]
    ax.bar(labels, values, yerr=errors, capsize=6,
           color=["tab:red", "tab:blue", "tab:green"], alpha=0.85)
    ax.set_ylim(min(values) - 4, max(values) + 3)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Tuning gained {tuned_gain:+.2f} pts, noise is ±{fold_std:.2f}")
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(values):
        ax.text(i, v + 0.4, f"{v:.2f}%", ha="center", fontsize=9)

    fig.suptitle("Day 13: Was Day 05's 80.9% real, or a lucky split?", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "cross_validation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")
    print(f"Saved {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
