"""
Day 15: A real land cover map of Abu Dhabi, and why random splits lie.

Day 13 ended with a conclusion: the features were the ceiling, not the
forest. The obvious test is to give the model the bands RGB was missing.
So this classifies a real Sentinel-2 scene using all ten useful bands
plus four spectral indices, instead of 17 colour statistics off a JPEG.

Labels come from ESA WorldCover, a global 10 m land cover product. That
makes this a supervised problem over a real image rather than a tidy
benchmark of pre-cut chips.

The second half is the part that matters. Neighbouring pixels in an
image are nearly identical, so a random pixel split puts a pixel in
training and the pixel touching it in test. The model can score
beautifully by memorising, and the number means nothing. Splitting by
spatial blocks instead gives the honest answer, and the gap between the
two is the subject of this day.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
from matplotlib.colors import ListedColormap
from odc.stac import load
from shapely.geometry import box, shape
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

# ---------------- Settings ----------------
BBOX = (54.36, 24.38, 54.56, 24.52)     # Abu Dhabi: mangroves, city, sea, desert
SEASON = "2024-01-01/2024-12-31"
MAX_CLOUD = 10
DAYS_PER_COMPOSITE = 3
CRS = "EPSG:32640"
RESOLUTION = 20                          # 20 m keeps the whole scene in memory

TRAIN_SAMPLES = 60000                    # subsample; the full scene is millions
BLOCKS = 6                               # split the scene into a 6x6 grid
MIN_CLASS_PIXELS = 2000                  # ignore classes too rare to learn
SEED = 42

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BAD_SCL = [0, 1, 3, 8, 9, 10]
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# ESA WorldCover class codes
WORLDCOVER = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare / sparse", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}


def catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def clear_days(client):
    """Days whose scenes cover the whole box, clearest first.

    The footprint check is from Day 03. A day that only half-covers the
    box produces a composite with a hole in it, and the hole is invisible
    in the summary statistics.
    """
    items = list(client.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=SEASON,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    ).items())
    if not items:
        raise RuntimeError("No Sentinel-2 scenes found")

    aoi = box(*BBOX)
    days = {}
    for it in items:
        days.setdefault(it.datetime.date(), []).append(it)

    usable = []
    for day, day_items in days.items():
        footprint = unary_union([shape(i.geometry) for i in day_items])
        if footprint.intersection(aoi).area / aoi.area < 0.99:
            continue
        cloud = sum(i.properties["eo:cloud_cover"] for i in day_items) / len(day_items)
        usable.append((cloud, day, day_items))

    if not usable:
        raise RuntimeError("No single day covers the whole box")
    usable.sort(key=lambda u: u[0])
    return usable[:DAYS_PER_COMPOSITE]


def reflectance(band, items):
    """Raw DN to surface reflectance, removing the baseline 04.00 offset."""
    offset = 1000 if any(
        float(i.properties.get("s2:processing_baseline", "0")) >= 4.0
        for i in items) else 0
    nodata = band == 0
    band = band.astype("float32") - offset
    return (band / 10000.0).clip(0, 1).where(~nodata)


def build_composite(client):
    """Median reflectance across several clear days, per band."""
    chosen = clear_days(client)
    print(f"  composite from {len(chosen)} clear day(s):")

    stacks = {b: [] for b in S2_BANDS}
    for cloud, day, items in chosen:
        print(f"    {day}  cloud {cloud:5.2f}%")
        ds = load(items, bands=S2_BANDS + ["SCL"], bbox=BBOX, crs=CRS,
                  resolution=RESOLUTION, groupby="solar_day").isel(time=0)
        clear = ~ds["SCL"].isin(BAD_SCL)
        for b in S2_BANDS:
            stacks[b].append(reflectance(ds[b], items).where(clear).values)

    with np.errstate(invalid="ignore"):
        return {b: np.nanmedian(np.stack(v), axis=0) for b, v in stacks.items()}


def spectral_indices(bands):
    """The four indices this course has already used, plus a built-up one."""
    def nd(a, b):
        denom = a + b
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denom > 0, (a - b) / denom, np.nan)

    return {
        "NDVI": nd(bands["B08"], bands["B04"]),      # vegetation, Day 01
        "NDWI": nd(bands["B03"], bands["B08"]),      # water, Day 03
        "MNDWI": nd(bands["B03"], bands["B11"]),     # water vs built, Day 03
        "NDBI": nd(bands["B11"], bands["B08"]),      # built-up
    }


def load_labels(client, shape_hw):
    """ESA WorldCover, the reference map used as training labels."""
    items = list(client.search(collections=["esa-worldcover"], bbox=BBOX).items())
    if not items:
        raise RuntimeError("No ESA WorldCover tile found for this box")
    newest = max(items, key=lambda i: i.properties.get("start_datetime", ""))
    print(f"  WorldCover tile {newest.id}")

    ds = load([newest], bands=["map"], bbox=BBOX, crs=CRS,
              resolution=RESOLUTION).isel(time=0)
    labels = ds["map"].values.astype(np.int16)
    if labels.shape != shape_hw:
        raise RuntimeError(f"label grid {labels.shape} does not match "
                           f"image grid {shape_hw}")
    return labels


def block_index(shape_hw, blocks=BLOCKS):
    """Assign every pixel to one of blocks x blocks spatial tiles."""
    h, w = shape_hw
    row = (np.arange(h) * blocks // h)[:, None]
    col = (np.arange(w) * blocks // w)[None, :]
    return (row * blocks + col).astype(np.int16)


def self_test():
    """Check the split logic on a synthetic image with known structure.

    The point being tested is the whole thesis of the day: a random pixel
    split on a spatially smooth image leaks, and a block split does not.
    So build an image whose label depends ONLY on position and not on the
    features at all. A model cannot legitimately learn anything from it,
    so any accuracy above chance is leakage.
    """
    rng = np.random.default_rng(0)
    h = w = 120

    # Labels in four spatial quadrants. Features are PURE NOISE.
    labels = np.zeros((h, w), dtype=int)
    labels[:h // 2, :w // 2] = 0
    labels[:h // 2, w // 2:] = 1
    labels[h // 2:, :w // 2] = 2
    labels[h // 2:, w // 2:] = 3

    # Smooth noise: neighbouring pixels are similar, like a real image
    coarse = rng.normal(size=(h // 6, w // 6, 4))
    features = np.repeat(np.repeat(coarse, 6, axis=0), 6, axis=1)
    features += rng.normal(0, 0.05, features.shape)

    X = features.reshape(-1, 4)
    y = labels.ravel()
    blocks = block_index((h, w), blocks=4).ravel()

    model = RandomForestClassifier(n_estimators=40, random_state=0, n_jobs=-1)

    idx = rng.permutation(len(X))
    cut = len(X) // 2
    model.fit(X[idx[:cut]], y[idx[:cut]])
    random_acc = accuracy_score(y[idx[cut:]], model.predict(X[idx[cut:]]))

    held = blocks >= 8
    model.fit(X[~held], y[~held])
    block_acc = accuracy_score(y[held], model.predict(X[held]))

    print(f"  features are pure noise, labels depend only on position:")
    print(f"    random pixel split -> {100 * random_acc:5.1f}% "
          f"(nothing real was learnable)")
    print(f"    spatial block split -> {100 * block_acc:5.1f}%")
    assert random_acc > 0.7, "the leak should be obvious on a random split"
    assert block_acc < random_acc, "a block split should expose the leak"
    print(f"    the {100 * (random_acc - block_acc):.0f} point gap is pure leakage")


def main():
    print("Self-test of the splitting logic:")
    self_test()

    client = catalog()
    print("\nBuilding the Sentinel-2 composite...")
    bands = build_composite(client)
    shape_hw = bands["B04"].shape
    print(f"  scene: {shape_hw[1]} x {shape_hw[0]} px at {RESOLUTION} m")

    print("\nLoading ESA WorldCover labels...")
    labels = load_labels(client, shape_hw)

    indices = spectral_indices(bands)
    feature_names = S2_BANDS + list(indices)
    stack = np.dstack([bands[b] for b in S2_BANDS] +
                      [indices[i] for i in indices])

    labelled = labels > 0
    finite = np.isfinite(stack).all(axis=2)
    valid = finite & labelled
    blocks = block_index(shape_hw)

    # A pixel is dropped if ANY feature is non-finite, which includes a
    # spectral index whose denominator came out at zero. That is easy to do
    # to a whole class at once: deep water has very low SWIR and NIR, so
    # NDBI can go undefined across every water pixel in the scene and the
    # class simply disappears from the results without any error.
    print("\n  Pixels dropped for a non-finite feature, by class:")
    silent_loss = []
    for code in sorted(WORLDCOVER):
        total = int((labels == code).sum())
        if total < MIN_CLASS_PIXELS:
            continue
        kept = int((valid & (labels == code)).sum())
        lost = 100 * (1 - kept / total)
        if lost > 0.5:
            print(f"    {WORLDCOVER[code][0]:<20} lost {lost:5.1f}% "
                  f"({total - kept:,} of {total:,})")
        if lost > 20:
            silent_loss.append(WORLDCOVER[code][0])
    if not silent_loss:
        print("    nothing lost more than 20%")
    else:
        print(f"    WARNING: {', '.join(silent_loss)} lost more than 20% of "
              f"their pixels.\n    Check the index denominators before "
              f"trusting any per-class number.")

    present = [(c, int((labels[valid] == c).sum())) for c in WORLDCOVER]
    present = [(c, n) for c, n in present if n >= MIN_CLASS_PIXELS]
    keep_classes = {c for c, _ in present}
    codes = sorted(keep_classes)
    valid &= np.isin(labels, list(keep_classes))

    print(f"\nScene composition ({int(valid.sum()):,} usable pixels, "
          f"{RESOLUTION} m):")
    for c, n in sorted(present, key=lambda p: -p[1]):
        print(f"  {WORLDCOVER[c][0]:<20} {n:9,}  "
              f"{100 * n / valid.sum():5.1f}%")

    X_all = stack[valid]
    y_all = labels[valid]
    b_all = blocks[valid]

    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X_all), min(TRAIN_SAMPLES * 2, len(X_all)),
                        replace=False)
    X, y, b = X_all[sample], y_all[sample], b_all[sample]

    model = lambda: RandomForestClassifier(n_estimators=200, random_state=SEED,
                                           n_jobs=-1)

    # ---------- The dishonest split ----------
    idx = rng.permutation(len(X))
    cut = len(X) // 2
    m = model().fit(X[idx[:cut]], y[idx[:cut]])
    random_pred = m.predict(X[idx[cut:]])
    random_acc = accuracy_score(y[idx[cut:]], random_pred)

    # ---------- The honest split ----------
    held_blocks = set(rng.choice(BLOCKS * BLOCKS, BLOCKS * BLOCKS // 3,
                                 replace=False).tolist())
    held = np.isin(b, list(held_blocks))
    m_block = model().fit(X[~held], y[~held])
    block_pred = m_block.predict(X[held])
    block_acc = accuracy_score(y[held], block_pred)

    print(f"\n--- The same model, two ways of splitting ---")
    print(f"  Random pixel split : {100 * random_acc:.2f}%  "
          f"({cut:,} train / {len(X) - cut:,} test)")
    print(f"  Spatial block split: {100 * block_acc:.2f}%  "
          f"({int((~held).sum()):,} train / {int(held.sum()):,} test, "
          f"{len(held_blocks)} of {BLOCKS * BLOCKS} blocks held out)")
    print(f"  Gap: {100 * (random_acc - block_acc):.2f} points of leakage")
    print(f"\n  A random split puts a pixel in training and the pixel touching "
          f"it in test.\n  At {RESOLUTION} m those two are nearly the same "
          f"measurement, so the model can\n  score well by memorising rather "
          f"than generalising.")

    # ---------- Day 13's question, answered ----------
    print(f"\n--- Did the extra bands help? ---")
    rgb_only = [feature_names.index(b) for b in ("B02", "B03", "B04")]
    m_rgb = model().fit(X[~held][:, rgb_only], y[~held])
    rgb_acc = accuracy_score(y[held], m_rgb.predict(X[held][:, rgb_only]))
    print(f"  RGB only, block split      : {100 * rgb_acc:.2f}%")
    print(f"  All {len(feature_names)} features, block split: "
          f"{100 * block_acc:.2f}%")
    print(f"  Day 13 predicted the features were the ceiling. Adding the "
          f"infrared\n  bands and indices is worth "
          f"{100 * (block_acc - rgb_acc):+.2f} points on the honest split.")

    # ---------- Overall accuracy is the wrong headline here ----------
    # Three classes make up most of the scene and they are the easy ones, so
    # overall accuracy largely reports "did it find the water". Balanced
    # accuracy averages recall over classes instead, giving a rare class the
    # same vote as a common one.
    balanced = balanced_accuracy_score(y[held], block_pred)
    macro_f1 = f1_score(y[held], block_pred, average="macro")

    print(f"\n--- Overall accuracy is flattering ---")
    print(f"  Overall accuracy  : {100 * block_acc:.2f}%  "
          f"(weighted by how common each class is)")
    print(f"  Balanced accuracy : {100 * balanced:.2f}%  "
          f"(every class counts the same)")
    print(f"  Macro F1          : {100 * macro_f1:.2f}%")

    counts = {c: int((y[held] == c).sum()) for c in codes}
    recalls = confusion_matrix(y[held], block_pred, labels=codes,
                               normalize="true").diagonal()
    print(f"\n  Per-class recall, held-out blocks:")
    for c, r in sorted(zip(codes, recalls), key=lambda p: -p[1]):
        bar = "#" * int(round(r * 30))
        print(f"    {WORLDCOVER[c][0]:<20} {100 * r:5.1f}%  "
              f"{counts[c]:7,} px  {bar}")

    big = [c for c in codes if counts[c] > 0.1 * len(y[held])]
    rare = [c for c in codes if c not in big]
    share = 100 * sum(counts[c] for c in big) / len(y[held])
    recall_of = dict(zip(codes, recalls))

    print(f"\n  {len(big)} classes hold {share:.0f}% of the test pixels.")
    if rare:
        big_mean = 100 * np.mean([recall_of[c] for c in big])
        rare_mean = 100 * np.mean([recall_of[c] for c in rare])
        print(f"  Mean recall on those common classes: {big_mean:.1f}%")
        print(f"  Mean recall on the {len(rare)} rare ones:      {rare_mean:.1f}%")
        if big_mean - rare_mean > 10:
            worst = min(rare, key=lambda c: recall_of[c])
            row = confusion_matrix(y[held], block_pred, labels=codes,
                                   normalize="true")[codes.index(worst)]
            taken_by = codes[int(np.argmax(np.where(
                np.arange(len(codes)) == codes.index(worst), -1, row)))]
            print(f"  The rare classes are largely collapsed into whichever "
                  f"common class they\n  most resemble: {WORLDCOVER[worst][0]} "
                  f"is mostly predicted as "
                  f"{WORLDCOVER[taken_by][0]}.")

    importance = sorted(zip(feature_names, m_block.feature_importances_),
                        key=lambda p: -p[1])
    print(f"\n  Most useful features: " +
          ", ".join(f"{n} ({100 * v:.1f}%)" for n, v in importance[:5]))

    # ---------- The map ----------
    predicted = np.full(shape_hw, 0, dtype=np.int16)
    predicted[valid] = m_block.predict(X_all)

    lut = {c: i + 1 for i, c in enumerate(codes)}
    cmap = ListedColormap(["#ffffff"] + [WORLDCOVER[c][1] for c in codes])

    def to_index(arr):
        out = np.zeros(arr.shape, dtype=np.int16)
        for c, i in lut.items():
            out[arr == c] = i
        return out

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    rgb = np.dstack([bands["B04"], bands["B03"], bands["B02"]])
    scale = np.nanpercentile(rgb, 98)
    axes[0, 0].imshow(np.nan_to_num(np.clip(rgb / max(scale, 1e-6), 0, 1)))
    axes[0, 0].set_title("Sentinel-2 true colour composite")

    axes[0, 1].imshow(to_index(np.where(valid, labels, 0)), cmap=cmap,
                      vmin=0, vmax=len(codes))
    axes[0, 1].set_title("ESA WorldCover (the labels)")

    axes[1, 0].imshow(to_index(predicted), cmap=cmap, vmin=0, vmax=len(codes))
    axes[1, 0].set_title(f"Predicted ({100 * block_acc:.1f}% overall, "
                         f"{100 * balanced:.1f}% balanced)")

    ax = axes[1, 1]
    cm = confusion_matrix(y[held], block_pred, labels=codes, normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    names = [WORLDCOVER[c][0] for c in codes]
    ax.set_xticks(range(len(codes)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(codes)), names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix, spatial block split")
    for i in range(len(codes)):
        for j in range(len(codes)):
            if cm[i, j] > 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks([]); ax.set_yticks([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=WORLDCOVER[c][1])
               for c in codes]
    axes[1, 0].legend(handles, names, fontsize=7, loc="upper left",
                      framealpha=0.9)

    fig.suptitle("Day 15: Land cover classification of Abu Dhabi, "
                 "and the cost of an honest split", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "landcover_map.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
