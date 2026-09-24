"""
Day 01: NDVI over Abu Dhabi's Eastern Mangroves using Sentinel-2 L2A.

Steps:
1. Search Microsoft Planetary Computer for the clearest Sentinel-2 scene.
2. Load Blue, Green, Red, NIR and the Scene Classification Layer (SCL).
3. Convert raw digital numbers to surface reflectance.
4. Mask clouds and shadows using SCL.
5. Compute NDVI, plot it, and estimate the area of dense vegetation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
from odc.stac import load

# ---------------- Settings ----------------
BBOX = (54.38, 24.40, 54.52, 24.50)   # lon_min, lat_min, lon_max, lat_max
DATE_RANGE = "2024-01-01/2024-12-31"
MAX_CLOUD = 5                          # percent
VEG_THRESHOLD = 0.3                    # NDVI above this = dense vegetation
CRS = "EPSG:32640"                     # UTM zone 40N, covers the UAE
PIXEL_AREA_M2 = 10 * 10                # Sentinel-2 10 m pixels

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# SCL classes to drop: no data, saturated, cloud shadow, clouds, cirrus
BAD_SCL = [0, 1, 3, 8, 9, 10]


def find_clearest_scene():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No scenes found. Widen DATE_RANGE or raise MAX_CLOUD.")

    item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
    print(f"Found {len(items)} scenes.")
    print(f"Using {item.id}")
    print(f"  Date: {item.datetime.date()}  Cloud: {item.properties['eo:cloud_cover']:.2f}%")
    return item


def to_reflectance(band, item):
    """Raw DN -> surface reflectance (0 to 1).

    Since processing baseline 04.00 (Jan 2022), ESA adds +1000 to every
    L2A value. We remove it so old and new scenes are comparable.
    """
    nodata = band == 0
    band = band.astype("float32")
    baseline = float(item.properties.get("s2:processing_baseline", "0"))
    if baseline >= 4.0:
        band = band - 1000
    band = (band / 10000.0).clip(0, 1)
    return band.where(~nodata)


def stretch(img, low=2, high=98):
    """Percentile stretch so the RGB image is not too dark."""
    lo, hi = np.nanpercentile(img, [low, high])
    return np.clip((img - lo) / (hi - lo), 0, 1)


def main():
    item = find_clearest_scene()

    ds = load(
        [item],
        bands=["B02", "B03", "B04", "B08", "SCL"],
        bbox=BBOX,
        crs=CRS,
        resolution=10,
    ).isel(time=0)

    blue = to_reflectance(ds["B02"], item)
    green = to_reflectance(ds["B03"], item)
    red = to_reflectance(ds["B04"], item)
    nir = to_reflectance(ds["B08"], item)

    clear = ~ds["SCL"].isin(BAD_SCL)

    denom = nir + red
    ndvi = ((nir - red) / denom).where((denom > 0) & clear)

    veg = ndvi > VEG_THRESHOLD
    veg_area_km2 = int(veg.sum()) * PIXEL_AREA_M2 / 1e6
    valid_px = int(ndvi.notnull().sum())

    print(f"Image size: {ndvi.shape[1]} x {ndvi.shape[0]} px")
    print(f"Mean NDVI (valid pixels): {float(ndvi.mean()):.3f}")
    print(f"Dense vegetation (NDVI > {VEG_THRESHOLD}): {veg_area_km2:.2f} km2 "
          f"({100 * int(veg.sum()) / valid_px:.1f}% of valid area)")

    # ---------------- Plots ----------------
    rgb = np.dstack([stretch(red.values), stretch(green.values), stretch(blue.values)])
    rgb = np.nan_to_num(rgb)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    date = item.datetime.date()

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"True colour (Sentinel-2, {date})")

    im = axes[0, 1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0, 1].set_title("NDVI")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].hist(ndvi.values[~np.isnan(ndvi.values)], bins=100, color="seagreen")
    axes[1, 0].axvline(VEG_THRESHOLD, color="red", linestyle="--", label=f"Threshold {VEG_THRESHOLD}")
    axes[1, 0].set_title("NDVI distribution")
    axes[1, 0].set_xlabel("NDVI")
    axes[1, 0].set_ylabel("Pixel count")
    axes[1, 0].legend()

    axes[1, 1].imshow(veg, cmap="Greens")
    axes[1, 1].set_title(f"Dense vegetation mask: {veg_area_km2:.2f} km²")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.axis("off")

    fig.suptitle("Day 01: NDVI over Abu Dhabi Eastern Mangroves", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "ndvi_mangroves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
