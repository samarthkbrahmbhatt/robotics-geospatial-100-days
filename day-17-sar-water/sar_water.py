"""
Day 17: Mapping water with radar, and why its noise is not like optical noise.

Every geospatial day so far has used an optical sensor: sunlight reflecting
off a surface, measured in bands. Sentinel-1 is a different instrument
entirely. It is a radar that supplies its own illumination, so it works at
night and straight through cloud, and what it measures is not colour but
how much of its own pulse the surface bounces back.

That changes the problem. Water is not detected by an index here, it is
detected by physics: a calm water surface acts like a mirror, reflecting
the pulse away from the satellite instead of back to it, so water returns
almost nothing and appears black. No band maths required.

The complication is speckle. SAR noise is MULTIPLICATIVE, not additive,
and it is severe: a single pixel of fully developed speckle has a standard
deviation equal to its own mean. Treating it like optical noise does not
work, and the whole first half of this is about that.

Day 03 mapped the same coastline with optical indices. The last section
puts the two answers side by side.
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
BBOX = (55.08, 25.05, 55.24, 25.18)      # Dubai coast, the Day 03 box
SAR_SEARCH = "2024-01-01/2024-12-31"
OPTICAL_SEARCH = "2024-01-01/2024-12-31"
MAX_CLOUD = 5
CRS = "EPSG:32640"
RESOLUTION = 20

FILTER_WINDOW = 5                         # boxcar window for speckle reduction

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BAD_SCL = [0, 1, 3, 8, 9, 10]


def catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


# ---------------- Speckle ----------------
def to_db(intensity, floor=1e-6):
    """Linear intensity to decibels.

    SAR values span several orders of magnitude, and speckle is
    multiplicative, so a log scale turns that multiplicative noise into
    additive noise and makes the histogram something a threshold can work
    on. Everything downstream happens in dB for that reason.
    """
    return 10.0 * np.log10(np.maximum(intensity, floor))


def boxcar(image, window=FILTER_WINDOW):
    """Mean filter via an integral image, ignoring NaN.

    Averaging N independent looks cuts the speckle standard deviation by
    sqrt(N). That is the entire mechanism, and the self-test checks the
    square root actually holds.
    """
    filled = np.nan_to_num(image, nan=0.0)
    mask = np.isfinite(image).astype(np.float64)

    def windowed_sum(a):
        padded = np.pad(a, window // 2, mode="edge")
        cumulative = padded.cumsum(axis=0).cumsum(axis=1)
        cumulative = np.pad(cumulative, ((1, 0), (1, 0)))
        h, w = a.shape
        return (cumulative[window:window + h, window:window + w]
                - cumulative[:h, window:window + w]
                - cumulative[window:window + h, :w]
                + cumulative[:h, :w])

    total = windowed_sum(filled)
    count = windowed_sum(mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1e-9), np.nan)


def coefficient_of_variation(a):
    """Standard deviation over mean. For fully developed 1-look speckle
    in intensity this is 1.0, which is a hard number to check against."""
    a = a[np.isfinite(a)]
    return float(a.std() / a.mean())


def otsu_threshold(values, n_bins=256):
    """The threshold that best splits a histogram into two groups.

    Same method as Day 03. On a SAR dB histogram over a coastline the two
    groups are water and everything else, and they are far better
    separated than they were in the optical indices.
    """
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2

    weight_below = np.cumsum(counts)
    weight_above = np.cumsum(counts[::-1])[::-1]
    valid = (weight_below[:-1] > 0) & (weight_above[1:] > 0)

    mean_below = np.cumsum(counts * centres) / np.maximum(weight_below, 1)
    mean_above = (np.cumsum((counts * centres)[::-1])
                  / np.maximum(weight_above[::-1], 1))[::-1]

    between = (weight_below[:-1] * weight_above[1:]
               * (mean_below[:-1] - mean_above[1:]) ** 2)
    between = np.where(valid, between, 0)
    return float(centres[np.argmax(between)])


# ---------------- Data ----------------
def load_sar(client):
    """One Sentinel-1 GRD scene covering the box.

    GRD values are digital numbers, not calibrated sigma-nought. Converting
    them to dB gives a RELATIVE scale, which is all a threshold needs: water
    is darker than land by a wide margin whatever the absolute calibration.
    Quoting these as physical backscatter would be wrong.
    """
    items = list(client.search(collections=["sentinel-1-grd"], bbox=BBOX,
                               datetime=SAR_SEARCH).items())
    if not items:
        raise RuntimeError("No Sentinel-1 GRD scenes found for this box")

    aoi = box(*BBOX)
    covering = []
    for it in items:
        if "vv" not in it.assets:
            continue
        overlap = shape(it.geometry).intersection(aoi).area / aoi.area
        if overlap >= 0.99:
            covering.append(it)

    if not covering:
        raise RuntimeError("No single Sentinel-1 scene covers the whole box")

    item = sorted(covering, key=lambda i: i.datetime)[len(covering) // 2]
    print(f"  {len(covering)} scenes fully cover the box, using {item.id}")
    print(f"    date {item.datetime.date()}, "
          f"orbit {item.properties.get('sat:orbit_state', '?')}, "
          f"mode {item.properties.get('sar:instrument_mode', '?')}")

    bands = ["vv", "vh"] if "vh" in item.assets else ["vv"]
    ds = load([item], bands=bands, bbox=BBOX, crs=CRS,
              resolution=RESOLUTION).isel(time=0)

    out = {}
    for b in bands:
        arr = ds[b].values.astype("float64")
        arr[arr <= 0] = np.nan
        out[b] = arr
    return out, item


def load_optical(client):
    """A clear Sentinel-2 scene over the same box, for the comparison."""
    items = list(client.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=OPTICAL_SEARCH,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}}).items())
    if not items:
        raise RuntimeError("No clear Sentinel-2 scenes found")

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
        raise RuntimeError("No Sentinel-2 day fully covers the box")

    cloud, day, day_items = min(usable, key=lambda u: u[0])
    print(f"  Sentinel-2 {day}, cloud {cloud:.2f}%")

    ds = load(day_items, bands=["B03", "B11", "SCL"], bbox=BBOX, crs=CRS,
              resolution=RESOLUTION, groupby="solar_day").isel(time=0)

    offset = 1000 if any(
        float(i.properties.get("s2:processing_baseline", "0")) >= 4.0
        for i in day_items) else 0

    def reflectance(band):
        nodata = band == 0
        band = band.astype("float32") - offset
        return (band / 10000.0).clip(0, 1).where(~nodata)

    green = reflectance(ds["B03"])
    swir = reflectance(ds["B11"])
    clear = ~ds["SCL"].isin(BAD_SCL)

    denom = green + swir
    mndwi = ((green - swir) / denom).where((denom > 0) & clear)
    return mndwi.values, day


def self_test():
    """Check the speckle maths against numbers theory fixes exactly."""
    rng = np.random.default_rng(0)

    # Fully developed speckle in intensity is exponentially distributed.
    # Its standard deviation equals its mean, so CV = 1. That is not an
    # approximation, it is the definition of 1-look speckle.
    truth = np.full((600, 600), 0.4)
    speckled = truth * rng.exponential(1.0, truth.shape)
    cv = coefficient_of_variation(speckled)
    print(f"  simulated 1-look speckle: coefficient of variation = {cv:.3f} "
          f"(theory says 1.000)")
    assert abs(cv - 1.0) < 0.05, "1-look speckle should have CV = 1"

    # Averaging N independent looks should cut CV by sqrt(N). A w x w
    # boxcar averages w^2 samples, so CV should fall by w.
    for window in (3, 5, 9):
        filtered = boxcar(speckled, window)
        measured = coefficient_of_variation(filtered)
        expected = 1.0 / window
        print(f"  {window}x{window} boxcar: CV = {measured:.3f}, "
              f"theory says {expected:.3f}")
        assert abs(measured - expected) < 0.25 * expected, \
            f"{window}x{window} filtering did not follow the square root law"

    # The filter must not invent signal: the mean has to survive
    assert abs(np.nanmean(boxcar(speckled, 5)) - speckled.mean()) < 0.01, \
        "the boxcar filter changed the mean"
    print(f"  boxcar preserves the mean (no signal invented)")

    # dB conversion must be monotonic and reversible
    values = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
    db = to_db(values)
    assert np.all(np.diff(db) > 0), "dB conversion must preserve ordering"
    assert np.allclose(10 ** (db / 10), values), "dB conversion is not reversible"
    print(f"  dB conversion: monotonic and reversible")

    # Otsu must find the valley between two known populations
    bimodal = np.concatenate([rng.normal(-18, 2, 20000),
                              rng.normal(-6, 2, 20000)])
    t = otsu_threshold(bimodal)
    print(f"  Otsu on two known peaks at -18 and -6 dB: threshold "
          f"{t:+.2f} dB")
    assert -14 < t < -10, "Otsu did not land between the two peaks"


def main():
    print("Self-test of the speckle and threshold maths:")
    self_test()

    client = catalog()

    print("\nLoading Sentinel-1 GRD...")
    sar, item = load_sar(client)
    vv = sar["vv"]

    vv_db = to_db(vv)
    vv_filtered_db = to_db(boxcar(vv, FILTER_WINDOW))

    print(f"  scene: {vv.shape[1]} x {vv.shape[0]} px at {RESOLUTION} m")

    # ---------- How bad is speckle, really? ----------
    print(f"\n--- Speckle ---")
    raw_cv = coefficient_of_variation(vv)
    filtered_cv = coefficient_of_variation(boxcar(vv, FILTER_WINDOW))
    print(f"  Coefficient of variation, whole scene:")
    print(f"    unfiltered      {raw_cv:.3f}")
    print(f"    {FILTER_WINDOW}x{FILTER_WINDOW} boxcar     "
          f"{filtered_cv:.3f}  ({raw_cv / max(filtered_cv, 1e-9):.1f}x "
          f"less variable)")
    print(f"  Note the whole-scene figure mixes real variation between "
          f"surfaces with\n  speckle, so it sits above the 1.0 that pure "
          f"speckle would give.")

    # ---------- Threshold ----------
    threshold = otsu_threshold(vv_filtered_db.ravel())
    water_sar = vv_filtered_db < threshold
    water_raw = vv_db < otsu_threshold(vv_db.ravel())

    pixel_area_km2 = (RESOLUTION ** 2) / 1e6
    print(f"\n--- Water from radar ---")
    print(f"  Otsu threshold: {threshold:+.2f} dB")
    print(f"  Water area, filtered:   "
          f"{np.nansum(water_sar) * pixel_area_km2:7.2f} km2")
    print(f"  Water area, unfiltered: "
          f"{np.nansum(water_raw) * pixel_area_km2:7.2f} km2")
    print(f"  Water is dark in radar because a calm surface reflects the "
          f"pulse away\n  from the satellite. That is physics, not an index.")

    # ---------- Compare against the optical answer ----------
    print(f"\nLoading Sentinel-2 for comparison...")
    mndwi, optical_day = load_optical(client)

    # The two collections are loaded separately, so their grids can differ
    # by a pixel from rounding even with the same bbox, CRS and resolution.
    # Crop both to the common extent rather than failing, but refuse if the
    # mismatch is large enough to mean they are not the same place.
    if mndwi.shape != vv.shape:
        h = min(mndwi.shape[0], vv.shape[0])
        w = min(mndwi.shape[1], vv.shape[1])
        drift = max(abs(mndwi.shape[0] - vv.shape[0]),
                    abs(mndwi.shape[1] - vv.shape[1]))
        if drift > 3:
            raise RuntimeError(
                f"optical grid {mndwi.shape} and SAR grid {vv.shape} differ by "
                f"{drift} pixels, which is too much to be rounding. The two "
                f"are not covering the same ground.")
        print(f"  grids differ by {drift} px from rounding, cropping both "
              f"to {h} x {w}")
        mndwi = mndwi[:h, :w]
        vv_filtered_db = vv_filtered_db[:h, :w]
        vv_db = vv_db[:h, :w]
        vv = vv[:h, :w]
        water_sar = water_sar[:h, :w]

    water_optical = mndwi > otsu_threshold(mndwi.ravel())
    both_valid = np.isfinite(vv_filtered_db) & np.isfinite(mndwi)

    agree = (water_sar == water_optical) & both_valid
    sar_only = water_sar & ~water_optical & both_valid
    optical_only = water_optical & ~water_sar & both_valid

    print(f"\n--- Radar against optical, same coastline ---")
    print(f"  Optical water: {np.nansum(water_optical & both_valid) * pixel_area_km2:7.2f} km2")
    print(f"  Radar water:   {np.nansum(water_sar & both_valid) * pixel_area_km2:7.2f} km2")
    print(f"  They agree on {100 * agree.sum() / both_valid.sum():.2f}% of pixels")
    print(f"    radar only:   {sar_only.sum() * pixel_area_km2:6.2f} km2")
    print(f"    optical only: {optical_only.sum() * pixel_area_km2:6.2f} km2")
    print(f"\n  Two instruments measuring completely different physics, "
          f"months apart,\n  on different orbits. Where they disagree is "
          f"more interesting than where\n  they agree: wet sand, calm pools, "
          f"and anything that changed in between.")

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    lo, hi = np.nanpercentile(vv_db, [2, 98])

    im = axes[0, 0].imshow(vv_db, cmap="gray", vmin=lo, vmax=hi)
    axes[0, 0].set_title(f"Sentinel-1 VV, raw ({item.datetime.date()})")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, label="dB (relative)")

    im = axes[0, 1].imshow(vv_filtered_db, cmap="gray", vmin=lo, vmax=hi)
    axes[0, 1].set_title(f"After a {FILTER_WINDOW}x{FILTER_WINDOW} boxcar")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, label="dB (relative)")

    ax = axes[0, 2]
    ax.hist(vv_db[np.isfinite(vv_db)], bins=150, alpha=0.5,
            color="tab:gray", label="Raw", density=True)
    ax.hist(vv_filtered_db[np.isfinite(vv_filtered_db)], bins=150, alpha=0.6,
            color="tab:blue", label="Filtered", density=True)
    ax.axvline(threshold, color="tab:red", linestyle="--",
               label=f"Otsu {threshold:+.1f} dB")
    ax.set_xlabel("VV (dB, relative)"); ax.set_ylabel("Density")
    ax.set_title("Filtering sharpens the two peaks")
    ax.legend(fontsize=8)

    axes[1, 0].imshow(water_sar, cmap="Blues")
    axes[1, 0].set_title(f"Radar water: "
                         f"{np.nansum(water_sar) * pixel_area_km2:.1f} km²")

    axes[1, 1].imshow(water_optical, cmap="Blues")
    axes[1, 1].set_title(f"Optical MNDWI water ({optical_day}): "
                         f"{np.nansum(water_optical) * pixel_area_km2:.1f} km²")

    comparison = np.zeros(vv.shape + (3,))
    comparison[...] = 0.95
    comparison[agree & water_sar] = (0.15, 0.35, 0.75)      # both say water
    comparison[agree & ~water_sar] = (0.93, 0.90, 0.82)     # both say land
    comparison[sar_only] = (0.9, 0.3, 0.1)                  # radar only
    comparison[optical_only] = (0.1, 0.7, 0.3)              # optical only
    axes[1, 2].imshow(comparison)
    axes[1, 2].set_title("Orange = radar only, green = optical only")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]):
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Day 17: Mapping water with radar, and living with speckle",
                 fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "sar_water.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
