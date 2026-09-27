"""
Day 03: Mapping water along the Dubai coastline with NDWI and MNDWI.

Two water indices, one automatic threshold (Otsu), and a shoreline
traced on top of the true colour image.

NDWI  = (Green - NIR)  / (Green + NIR)
MNDWI = (Green - SWIR) / (Green + SWIR)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
from shapely.geometry import box, shape
from shapely.ops import unary_union
from odc.stac import load

# ---------------- Settings ----------------
BBOX = (55.08, 25.05, 55.24, 25.18)   # Palm Jumeirah, Marina, Dubai coast
DATE_RANGE = "2024-01-01/2024-12-31"
MAX_CLOUD = 5
CRS = "EPSG:32640"
PIXEL_AREA_M2 = 10 * 10

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BAD_SCL = [0, 1, 3, 8, 9, 10]   # nodata, saturated, shadow, clouds, cirrus


def find_best_day():
    """Pick the clearest day whose scenes actually cover the whole box.

    Scenes from a single day all come from one orbit pass, and the box
    can stick out past the edge of that pass. Comparing each day's
    footprint against the box first avoids downloading a half-empty
    mosaic.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    ).items())
    if not items:
        raise RuntimeError("No scenes found. Widen DATE_RANGE or raise MAX_CLOUD.")

    aoi = box(*BBOX)

    days = {}
    for it in items:
        days.setdefault(it.datetime.date(), []).append(it)

    summary = []
    for day, day_items in days.items():
        footprint = unary_union([shape(i.geometry) for i in day_items])
        covered = footprint.intersection(aoi).area / aoi.area
        cloud = sum(i.properties["eo:cloud_cover"] for i in day_items) / len(day_items)
        summary.append((day, day_items, covered, cloud))

    print(f"Found {len(items)} scenes across {len(days)} days.")

    full = [s for s in summary if s[2] >= 0.99]
    if full:
        day, day_items, covered, cloud = min(full, key=lambda s: s[3])
        print(f"{len(full)} day(s) fully cover the box. Picking the clearest.")
    else:
        day, day_items, covered, cloud = max(summary, key=lambda s: s[2])
        print("WARNING: no single day covers the whole box. Using the best available.")

    tiles = sorted({i.properties.get("s2:mgrs_tile", "?") for i in day_items})
    print(f"Using {day}: {len(day_items)} scene(s), tiles {', '.join(tiles)}")
    print(f"  Footprint covers {100 * covered:.1f}% of the box, mean cloud {cloud:.2f}%")
    return day_items, day


def to_reflectance(band, item):
    """Raw DN -> surface reflectance, removing the baseline 04.00 offset."""
    nodata = band == 0
    band = band.astype("float32")
    if float(item.properties.get("s2:processing_baseline", "0")) >= 4.0:
        band = band - 1000
    return (band / 10000.0).clip(0, 1).where(~nodata)


def normalised_difference(a, b):
    denom = a + b
    return ((a - b) / denom).where(denom > 0)


def otsu_threshold(values, n_bins=256):
    """Find the threshold that best splits values into two groups.

    Otsu's method: try every candidate cut and keep the one that
    maximises the variance *between* the two groups. Written out here
    rather than imported, because the logic is the point of the exercise.
    """
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2

    weight_below = np.cumsum(counts)
    weight_above = np.cumsum(counts[::-1])[::-1]

    # Guard against empty groups at the extreme ends
    valid = (weight_below[:-1] > 0) & (weight_above[1:] > 0)

    mean_below = np.cumsum(counts * centres) / np.maximum(weight_below, 1)
    mean_above = (np.cumsum((counts * centres)[::-1]) / np.maximum(weight_above[::-1], 1))[::-1]

    between_variance = (weight_below[:-1] * weight_above[1:]
                        * (mean_below[:-1] - mean_above[1:]) ** 2)
    between_variance = np.where(valid, between_variance, 0)

    return float(centres[np.argmax(between_variance)])


def water_area_km2(mask):
    return int(mask.sum()) * PIXEL_AREA_M2 / 1e6


def main():
    items, day = find_best_day()

    # groupby="solar_day" mosaics all tiles from the same day into one image
    ds = load(
        items,
        bands=["B02", "B03", "B04", "B08", "B11", "SCL"],
        bbox=BBOX,
        crs=CRS,
        resolution=10,
        groupby="solar_day",
    ).isel(time=0)

    item = items[0]   # processing baseline is shared across the day

    blue = to_reflectance(ds["B02"], item)
    green = to_reflectance(ds["B03"], item)
    red = to_reflectance(ds["B04"], item)
    nir = to_reflectance(ds["B08"], item)
    swir = to_reflectance(ds["B11"], item)

    clear = ~ds["SCL"].isin(BAD_SCL)

    ndwi = normalised_difference(green, nir).where(clear)
    mndwi = normalised_difference(green, swir).where(clear)

    t_ndwi = otsu_threshold(ndwi.values.ravel())
    t_mndwi = otsu_threshold(mndwi.values.ravel())

    water_ndwi = ndwi > t_ndwi
    water_mndwi = mndwi > t_mndwi

    coverage = 100 * float(np.isfinite(ndwi.values).mean())
    print(f"Image size: {ndwi.shape[1]} x {ndwi.shape[0]} px")
    print(f"Valid data coverage: {coverage:.1f}% of the box")
    if coverage < 95:
        print("  WARNING: parts of the box have no data. Check the true colour panel.")
    print(f"Otsu threshold  NDWI: {t_ndwi:+.3f}   MNDWI: {t_mndwi:+.3f}")
    print(f"Water area      NDWI: {water_area_km2(water_ndwi):.2f} km2   "
          f"MNDWI: {water_area_km2(water_mndwi):.2f} km2")

    disagree = water_ndwi != water_mndwi
    print(f"The two indices disagree on {100 * float(disagree.mean()):.2f}% of pixels")

    # ---------------- Plots ----------------
    def stretch(img, low=2, high=98):
        lo, hi = np.nanpercentile(img, [low, high])
        return np.clip((img - lo) / (hi - lo), 0, 1)

    rgb = np.nan_to_num(np.dstack([stretch(red.values),
                                   stretch(green.values),
                                   stretch(blue.values)]))

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"True colour ({day})")

    im = axes[0, 1].imshow(ndwi, cmap="BrBG", vmin=-1, vmax=1)
    axes[0, 1].set_title(f"NDWI (green vs NIR)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(mndwi, cmap="BrBG", vmin=-1, vmax=1)
    axes[0, 2].set_title("MNDWI (green vs SWIR)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    for data, thresh, name, colour in ((ndwi, t_ndwi, "NDWI", "tab:blue"),
                                       (mndwi, t_mndwi, "MNDWI", "tab:orange")):
        flat = data.values.ravel()
        ax.hist(flat[np.isfinite(flat)], bins=120, alpha=0.55, color=colour, label=name)
        ax.axvline(thresh, color=colour, linestyle="--",
                   label=f"{name} Otsu {thresh:+.2f}")
    ax.set_title("Index distributions and automatic thresholds")
    ax.set_xlabel("Index value")
    ax.set_ylabel("Pixel count")
    ax.legend(fontsize=8)

    axes[1, 1].imshow(water_mndwi, cmap="Blues")
    axes[1, 1].set_title(f"MNDWI water mask: {water_area_km2(water_mndwi):.1f} km²")

    axes[1, 2].imshow(rgb)
    axes[1, 2].contour(mndwi.values, levels=[t_mndwi], colors="red", linewidths=0.6)
    axes[1, 2].set_title("Extracted shoreline on true colour")

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]):
        ax.axis("off")

    fig.suptitle("Day 03: Water mapping on the Dubai coastline", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "water_mapping.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
