"""
Day 07: Mangrove change detection, Abu Dhabi, 2018 vs 2024.

Day 01 computed NDVI for one date. One date is not enough to say
anything changed: tide, season and haze all move NDVI on their own.

So this builds a median composite from several clear days in the same
months of each year, differences the two composites, and reports how
much vegetation was gained and lost.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
from odc.stac import load
from shapely.geometry import box, shape
from shapely.ops import unary_union

# ---------------- Settings ----------------
BBOX = (54.38, 24.40, 54.52, 24.50)      # Abu Dhabi Eastern Mangroves
SEASONS = {2018: "2018-10-01/2019-03-31",
           2024: "2024-10-01/2025-03-31"}
MAX_CLOUD = 10
DAYS_PER_COMPOSITE = 4
CHANGE_THRESHOLD = 0.15                  # NDVI shift counted as real change
VEG_THRESHOLD = 0.3
CRS = "EPSG:32640"
PIXEL_AREA_M2 = 10 * 10

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BAD_SCL = [0, 1, 3, 8, 9, 10]


def catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def clear_days(client, date_range):
    """Days whose scenes together cover the whole box, clearest first."""
    items = list(client.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=date_range,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    ).items())
    if not items:
        raise RuntimeError(f"No scenes for {date_range}")

    aoi = box(*BBOX)
    days = {}
    for it in items:
        days.setdefault(it.datetime.date(), []).append(it)

    usable = []
    for day, day_items in days.items():
        footprint = unary_union([shape(i.geometry) for i in day_items])
        if footprint.intersection(aoi).area / aoi.area < 0.99:
            continue                       # partial coverage, skip it
        cloud = sum(i.properties["eo:cloud_cover"] for i in day_items) / len(day_items)
        usable.append((cloud, day, day_items))

    if not usable:
        raise RuntimeError(f"No day fully covers the box in {date_range}")

    usable.sort(key=lambda u: u[0])
    return usable[:DAYS_PER_COMPOSITE]


def ndvi_for_day(day_items):
    ds = load(day_items, bands=["B04", "B08", "SCL"], bbox=BBOX,
              crs=CRS, resolution=10, groupby="solar_day").isel(time=0)

    baseline = float(day_items[0].properties.get("s2:processing_baseline", "0"))

    def reflectance(band):
        nodata = band == 0
        band = band.astype("float32")
        if baseline >= 4.0:
            band = band - 1000
        return (band / 10000.0).clip(0, 1).where(~nodata)

    red, nir = reflectance(ds["B04"]), reflectance(ds["B08"])
    clear = ~ds["SCL"].isin(BAD_SCL)
    denom = nir + red
    return ((nir - red) / denom).where((denom > 0) & clear).values


def composite(client, year):
    """Median NDVI across several clear days, so one odd day cannot dominate."""
    chosen = clear_days(client, SEASONS[year])
    print(f"\n{year} season: using {len(chosen)} day(s)")
    stack = []
    for cloud, day, day_items in chosen:
        print(f"  {day}  cloud {cloud:5.2f}%  ({len(day_items)} scene(s))")
        stack.append(ndvi_for_day(day_items))

    arr = np.stack(stack)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(arr, axis=0)
    coverage = 100 * np.isfinite(median).mean()
    print(f"  Composite valid on {coverage:.1f}% of pixels")
    return median


def area_km2(mask):
    return int(np.nansum(mask)) * PIXEL_AREA_M2 / 1e6


def main():
    client = catalog()
    early = composite(client, 2018)
    late = composite(client, 2024)

    diff = late - early
    valid = np.isfinite(diff)

    gain = valid & (diff > CHANGE_THRESHOLD)
    loss = valid & (diff < -CHANGE_THRESHOLD)
    stable = valid & ~gain & ~loss

    veg_early = valid & (early > VEG_THRESHOLD)
    veg_late = valid & (late > VEG_THRESHOLD)

    print("\n--- Change 2018 to 2024 ---")
    print(f"Dense vegetation (NDVI > {VEG_THRESHOLD}):")
    print(f"  2018: {area_km2(veg_early):6.2f} km2")
    print(f"  2024: {area_km2(veg_late):6.2f} km2")
    print(f"  Net:  {area_km2(veg_late) - area_km2(veg_early):+6.2f} km2")
    print(f"\nPixel-level change (NDVI shift > {CHANGE_THRESHOLD}):")
    print(f"  Gain:   {area_km2(gain):6.2f} km2 ({100 * gain.sum() / valid.sum():4.1f}% of valid area)")
    print(f"  Loss:   {area_km2(loss):6.2f} km2 ({100 * loss.sum() / valid.sum():4.1f}%)")
    print(f"  Stable: {area_km2(stable):6.2f} km2 ({100 * stable.sum() / valid.sum():4.1f}%)")
    print(f"\nMean NDVI shift across valid pixels: {np.nanmean(diff):+.3f}")

    # --- How much of that "gain" is actually vegetation? ---
    # A pixel going from open water (NDVI well below zero) to exposed mudflat
    # (NDVI near zero) clears the +0.15 threshold without anything growing.
    # That is a tide artefact, not greening.
    was_water = valid & (early < -0.10)
    gain_from_water = gain & was_water
    gain_to_vegetation = gain & veg_late

    print("\nBreaking the gain down:")
    print(f"  Started as open water (tide artefact): {area_km2(gain_from_water):6.2f} km2 "
          f"({100 * gain_from_water.sum() / max(gain.sum(), 1):4.1f}% of all gain)")
    print(f"  Ended above the vegetation threshold:  {area_km2(gain_to_vegetation):6.2f} km2 "
          f"({100 * gain_to_vegetation.sum() / max(gain.sum(), 1):4.1f}% of all gain)")
    print(f"  Water extent: {area_km2(was_water):6.2f} km2 in 2018, "
          f"{area_km2(valid & (late < -0.10)):6.2f} km2 in 2024")

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, data, title in ((axes[0, 0], early, "NDVI composite, 2018-19 season"),
                            (axes[0, 1], late, "NDVI composite, 2024-25 season")):
        im = ax.imshow(data, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(diff, cmap="bwr_r", vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title("NDVI change (blue = gain, red = loss)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    rgb = np.zeros(diff.shape + (3,))
    rgb[stable] = (0.92, 0.92, 0.92)
    rgb[gain] = (0.1, 0.5, 0.15)
    rgb[loss] = (0.75, 0.1, 0.1)
    axes[1, 1].imshow(rgb)
    axes[1, 1].set_title(f"Gain {area_km2(gain):.2f} km²   |   Loss {area_km2(loss):.2f} km²")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("Day 07: Mangrove change, Abu Dhabi, 2018 to 2024", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "mangrove_change.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
