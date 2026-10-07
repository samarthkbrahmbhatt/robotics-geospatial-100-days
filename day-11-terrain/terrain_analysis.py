"""
Day 11: Terrain analysis of the Hajar mountains from the Copernicus DEM.

Every previous geospatial day used reflectance: how much light a surface
sends back. This one uses elevation, which is a different kind of raster
and needs different maths.

From a grid of heights, three derived products:

  slope      how steep each pixel is
  aspect     which compass direction it faces
  hillshade  what the terrain looks like lit from one direction

All three come from the same thing: the gradient of the elevation
surface. Getting that right depends on the pixel size being in metres,
which is why the DEM is reprojected to UTM before anything is computed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
from matplotlib.colors import LightSource
from odc.stac import load

# ---------------- Settings ----------------
# Jebel Jais and the surrounding Hajar range, the highest terrain in the UAE
BBOX = (56.00, 25.85, 56.35, 26.10)
CRS = "EPSG:32640"
RESOLUTION = 30          # metres, the native resolution of Copernicus GLO-30

SEA_LEVEL_CUTOFF = 5     # metres; below this is sea or coastal flat, not mountain
SUN_AZIMUTH = 315        # degrees, light from the northwest by convention
SUN_ALTITUDE = 45        # degrees above the horizon

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def slope_and_aspect(dem, pixel_size):
    """Slope in degrees and aspect as a compass bearing, from the gradient.

    np.gradient returns change per pixel, so passing the pixel size in
    metres makes it a real gradient (rise over run).

    Axis order matters and is easy to get wrong. np.gradient returns
    (change per row, change per column). In a north-up raster the row
    index increases *southward*, so the northward gradient is the
    negative of the row gradient.

    Aspect is the compass bearing of the downhill direction. The downhill
    vector has east component -dz_dx and north component -dz_dy, and a
    compass bearing is measured clockwise from north, which is
    atan2(east, north) rather than the usual atan2(y, x).
    """
    d_row, d_col = np.gradient(dem, pixel_size)
    dz_dx = d_col            # change per metre eastward
    dz_dy = -d_row           # change per metre northward

    gradient_magnitude = np.hypot(dz_dx, dz_dy)
    slope = np.degrees(np.arctan(gradient_magnitude))

    aspect = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0
    aspect[gradient_magnitude < 1e-9] = np.nan   # flat ground faces nowhere

    return slope, aspect


def self_test():
    """Check slope and aspect against surfaces with a known answer.

    A plane rising 1 m per metre travelled is exactly 45 degrees, and one
    rising 0.5 m per metre is arctan(0.5). Aspect is the *downhill*
    bearing, so a plane rising toward the east faces west, at 270 degrees.

    All four compass directions are tested. Testing only east and west
    would miss a sign error on the row axis, which flips north and south
    by 180 degrees while leaving east and west correct.
    """
    size, pixel = 50, 10.0
    ramp = np.arange(size) * pixel
    reversed_ramp = (size - 1 - np.arange(size)) * pixel

    # (name, elevation grid, expected downhill bearing)
    cases = [
        ("rises to the east",  np.tile(ramp, (size, 1)),                  270.0),
        ("rises to the west",  np.tile(reversed_ramp, (size, 1)),          90.0),
        ("rises to the north", np.tile(reversed_ramp[:, None], (1, size)), 180.0),
        ("rises to the south", np.tile(ramp[:, None], (1, size)),           0.0),
    ]

    expected_slope = np.degrees(np.arctan(1.0))
    for name, dem, expected_aspect in cases:
        slope, aspect = slope_and_aspect(dem, pixel)
        s = float(np.median(slope[1:-1, 1:-1]))
        a = float(np.median(aspect[1:-1, 1:-1])) % 360.0
        assert abs(s - expected_slope) < 1e-9, f"{name}: slope {s} != {expected_slope}"
        assert abs(a - expected_aspect) < 1e-9, f"{name}: aspect {a} != {expected_aspect}"
        print(f"  {name:<20} slope {s:7.3f} deg, faces {a:5.1f} deg "
              f"(expected {expected_aspect:5.1f})")

    # A gentler slope, to confirm the magnitude and not just the direction
    gentle = np.tile(ramp * 0.5, (size, 1))
    s = float(np.median(slope_and_aspect(gentle, pixel)[0][1:-1, 1:-1]))
    expected = np.degrees(np.arctan(0.5))
    assert abs(s - expected) < 1e-9, f"gentle slope {s} != {expected}"
    print(f"  {'half-gradient plane':<20} slope {s:7.3f} deg (expected {expected:.3f})")

    flat = np.zeros((size, size))
    slope, aspect = slope_and_aspect(flat, pixel)
    assert np.allclose(slope, 0), "flat ground should have zero slope"
    assert np.all(np.isnan(aspect)), "flat ground should have undefined aspect"
    print(f"  {'flat ground':<20} slope 0, aspect undefined")


def main():
    print("Self-test of the slope and aspect maths:")
    self_test()

    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(client.search(collections=["cop-dem-glo-30"], bbox=BBOX).items())
    if not items:
        raise RuntimeError("No DEM tiles found for this box")
    print(f"\nFound {len(items)} DEM tile(s)")

    ds = load(items, bands=["data"], bbox=BBOX, crs=CRS, resolution=RESOLUTION)
    dem = ds["data"].isel(time=0).values.astype("float32")
    dem[dem < -500] = np.nan                  # nodata only

    print(f"DEM: {dem.shape[1]} x {dem.shape[0]} px at {RESOLUTION} m")
    print(f"Elevation: {np.nanmin(dem):.0f} m to {np.nanmax(dem):.0f} m, "
          f"mean {np.nanmean(dem):.0f} m")

    slope, aspect = slope_and_aspect(dem, RESOLUTION)

    # The western edge of this box is the Arabian Gulf, which sits at about
    # 0 m and is perfectly flat. Leaving it in means every slope statistic
    # is an average over land AND sea, which understates the mountains.
    # Statistics are reported over land only; the maps still show everything.
    land = dem > SEA_LEVEL_CUTOFF
    sea_share = 100 * (1 - np.nansum(land) / np.sum(np.isfinite(dem)))
    print(f"Sea and near-sea-level ground (below {SEA_LEVEL_CUTOFF} m): "
          f"{sea_share:.1f}% of the box, excluded from the statistics below")

    land_slope = np.where(land, slope, np.nan)

    print(f"\nSlope over land: mean {np.nanmean(land_slope):.1f} deg, "
          f"median {np.nanmedian(land_slope):.1f} deg, max {np.nanmax(land_slope):.1f} deg")
    for threshold in (10, 20, 30, 40):
        share = 100 * np.nanmean(land_slope[np.isfinite(land_slope)] > threshold)
        everything = 100 * np.nanmean(slope > threshold)
        print(f"  steeper than {threshold:2d} deg: {share:5.1f}% of land "
              f"({everything:5.1f}% if the sea is counted too)")

    # Which way do the slopes face? Relevant for sun exposure and vegetation.
    print("\nAspect, counting only land steeper than 5 degrees:")
    steep = land & (slope > 5)
    sectors = {"North (315-45)": ((aspect >= 315) | (aspect < 45)),
               "East  (45-135)": ((aspect >= 45) & (aspect < 135)),
               "South (135-225)": ((aspect >= 135) & (aspect < 225)),
               "West  (225-315)": ((aspect >= 225) & (aspect < 315))}
    total = np.nansum(steep)
    for name, mask in sectors.items():
        print(f"  {name:<16} {100 * np.nansum(mask & steep) / total:5.1f}%")

    ls = LightSource(azdeg=SUN_AZIMUTH, altdeg=SUN_ALTITUDE)
    hillshade = ls.hillshade(np.nan_to_num(dem, nan=0.0),
                             dx=RESOLUTION, dy=RESOLUTION)

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    im = axes[0, 0].imshow(dem, cmap="terrain")
    axes[0, 0].set_title(f"Elevation ({np.nanmin(dem):.0f} to {np.nanmax(dem):.0f} m)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, label="metres")

    axes[0, 1].imshow(hillshade, cmap="gray")
    axes[0, 1].set_title(f"Hillshade (sun from {SUN_AZIMUTH} deg, "
                         f"{SUN_ALTITUDE} deg up)")

    im = axes[1, 0].imshow(slope, cmap="magma", vmin=0, vmax=min(60, np.nanmax(slope)))
    axes[1, 0].set_title(f"Slope (mean over land {np.nanmean(land_slope):.1f} deg)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, label="degrees")

    masked_aspect = np.where(steep, aspect, np.nan)
    im = axes[1, 1].imshow(masked_aspect, cmap="twilight", vmin=0, vmax=360)
    axes[1, 1].set_title("Aspect (compass bearing of the downhill direction)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, label="degrees",
                 ticks=[0, 90, 180, 270, 360])

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("Day 11: Terrain of the Hajar mountains from the Copernicus DEM",
                 fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "terrain_analysis.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
