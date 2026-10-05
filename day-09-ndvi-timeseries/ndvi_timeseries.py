"""
Day 09: A year of NDVI for four land cover types in the UAE.

Days 01, 03 and 07 each looked at one or two dates. This one adds the
time dimension properly: every usable Sentinel-2 pass over 2024 for four
sites, loaded as a single xarray cube per site.

Sites: mangrove, irrigated farmland, open desert, dense city.

A note on how the sites are defined. My first attempt pinned each site
to a single hand-picked coordinate, and two of the four landed on the
wrong thing: the "mangrove" point was open water and the "farmland"
point was dune. So the vegetated sites are now defined as a box plus a
rule: keep only pixels whose median NDVI across the whole year is above
0.25. That finds the vegetation inside the box wherever it actually sits,
instead of trusting a coordinate I guessed from a map.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
from odc.stac import load

# ---------------- Sites ----------------
# lon, lat, and whether to keep only persistently vegetated pixels
SITES = {
    "Mangrove (Abu Dhabi)":  (54.429, 24.455, True),
    "Farmland (Liwa)":       (53.780, 23.120, True),
    "Desert (Rub al Khali)": (54.500, 23.000, False),
    "City (Dubai)":          (55.270, 25.200, False),
}
BOX_DEG = 0.015              # about 3 km across
RESOLUTION = 20              # metres, keeps a year of data in memory comfortably
YEAR = "2024-01-01/2024-12-31"
MAX_CLOUD = 20
MIN_VALID_FRACTION = 0.6     # drop a date if most of the box is masked
VEG_MEDIAN = 0.25            # a pixel counts as vegetation if its yearly median clears this

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BAD_SCL = [0, 1, 3, 8, 9, 10]


def site_series(client, lon, lat, vegetated_only):
    bbox = (lon - BOX_DEG, lat - BOX_DEG, lon + BOX_DEG, lat + BOX_DEG)
    items = list(client.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=YEAR,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    ).items())
    if not items:
        raise RuntimeError(f"No scenes for {bbox}")

    # One call, all dates: the cube has a time dimension
    ds = load(items, bands=["B02", "B03", "B04", "B08", "SCL"],
              bbox=bbox, crs="EPSG:32640", resolution=RESOLUTION, groupby="solar_day")

    offset = 1000 if any(float(i.properties.get("s2:processing_baseline", "0")) >= 4.0
                         for i in items) else 0

    def reflectance(band):
        nodata = band == 0
        band = band.astype("float32") - offset
        return (band / 10000.0).clip(0, 1).where(~nodata)

    red, nir = reflectance(ds["B04"]), reflectance(ds["B08"])
    clear = ~ds["SCL"].isin(BAD_SCL)
    denom = nir + red
    ndvi = ((nir - red) / denom).where((denom > 0) & clear)

    keep = ndvi.notnull().mean(dim=("x", "y")) >= MIN_VALID_FRACTION
    ndvi = ndvi.where(keep, drop=True)

    if vegetated_only:
        yearly_median = ndvi.median(dim="time", skipna=True)
        mask = yearly_median > VEG_MEDIAN
        kept = float(mask.mean()) * 100
        if kept < 2:
            print(f"    WARNING: only {kept:.1f}% of this box is persistently vegetated. "
                  f"The box may be in the wrong place.")
        series = ndvi.where(mask).mean(dim=("x", "y"), skipna=True)
    else:
        kept = 100.0
        series = ndvi.mean(dim=("x", "y"), skipna=True)

    # True colour chip from the clearest date, so the site can be checked by eye
    best = int(np.argmax(ndvi.notnull().mean(dim=("x", "y")).values))
    rgb = np.dstack([reflectance(ds[b]).where(keep, drop=True).isel(time=best).values
                     for b in ("B04", "B03", "B02")])
    scale = np.nanpercentile(rgb, 98)
    chip = np.clip(rgb / (scale if scale > 0 else 1), 0, 1)

    return (pd.to_datetime(series.time.values), series.values,
            np.nan_to_num(chip), len(items), kept)


def main():
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    results = {}
    print("Loading a year of Sentinel-2 for each site...\n")
    for name, (lon, lat, vegetated_only) in SITES.items():
        dates, values, chip, n_items, kept = site_series(client, lon, lat, vegetated_only)
        results[name] = (dates, values, chip)
        extra = f", {kept:.0f}% of box kept as vegetation" if vegetated_only else ""
        print(f"  {name:<24} {n_items:3d} scenes, {len(values):3d} dates usable{extra}")

    print("\n--- A year of NDVI ---")
    amplitude = {}
    for name, (dates, values, _) in results.items():
        peak = dates[int(np.nanargmax(values))]
        low = dates[int(np.nanargmin(values))]
        # Min-to-max is driven by single odd dates. The 10th-90th percentile
        # range describes the year without letting one hazy pass define it.
        p10, p90 = np.nanpercentile(values, [10, 90])
        amplitude[name] = p90 - p10
        print(f"{name:<24} n={len(values):3d}  mean {np.nanmean(values):+.3f}  "
              f"min-max {np.nanmin(values):+.3f} to {np.nanmax(values):+.3f}  "
              f"p10-p90 spread {amplitude[name]:.3f}  peak {peak:%b}  low {low:%b}")

    print(f"\nLargest spread through the year: {max(amplitude, key=amplitude.get)}")
    print(f"Flattest: {min(amplitude, key=amplitude.get)}")

    # How much of the variation is day-to-day scatter rather than season?
    print("\nDay-to-day scatter (median change between consecutive passes):")
    for name, (dates, values, _) in results.items():
        jumps = np.abs(np.diff(values))
        gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
        quick = jumps[gaps <= 10]
        print(f"  {name:<24} {np.nanmedian(quick):.3f} NDVI within 10 days "
              f"(seasonal p10-p90 spread is {amplitude[name]:.3f})")

    means = {n: np.nanmean(v[1]) for n, v in results.items()}
    if means["Mangrove (Abu Dhabi)"] < 0.3:
        print("\nWARNING: mangrove NDVI is low for dense evergreen vegetation. Check the chip.")
    if means["Farmland (Liwa)"] < 0.2:
        print("WARNING: farmland NDVI is low. The box may not contain an active field.")
    if means["Desert (Rub al Khali)"] > 0.15:
        print("WARNING: the desert site is greener than expected. Check the chip.")

    # ---------------- Plots ----------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], hspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    for (name, (dates, values, _)), colour in zip(
            results.items(), ["tab:green", "tab:olive", "tab:orange", "tab:gray"]):
        ax.plot(dates, values, "o-", markersize=3, linewidth=1.4,
                color=colour, label=name, alpha=0.9)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_ylabel("NDVI (site mean)")
    ax.set_title("NDVI through 2024, four land cover types")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    for i, (name, (_, _, chip)) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(chip)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    fig.suptitle("Day 09: A year of NDVI, four UAE land cover types", fontsize=15)
    out = OUT_DIR / "ndvi_timeseries.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")
    print("Check the four chips against their labels before trusting the curves.")


if __name__ == "__main__":
    main()
