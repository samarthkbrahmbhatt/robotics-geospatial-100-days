"""
Day 19: Predicting land surface temperature, and why R-squared flatters.

Every geospatial day so far has been classification or thresholding: is
this pixel water, is it mangrove, which of ten classes is it. This one
predicts a continuous number instead, which brings a different set of
metrics and a different set of ways to be wrong.

The target is land surface temperature from Landsat's thermal band. The
predictors are Sentinel-2 reflectance from a nearby date. The physical
story is the urban heat island: vegetation cools itself by evaporating
water, so green pixels run cooler than concrete ones, and that
relationship is what the model has to find.

Two things get examined that a single accuracy number hides:

  1. R-squared depends on how variable the target is, not just on how
     good the model is. The same model scores differently on a scene
     with more spread, which makes it useless for comparing across
     studies.

  2. A regression fitted to minimise squared error systematically
     under-predicts hot pixels and over-predicts cold ones. The scatter
     of predicted against actual has a slope below 1, always, and that
     matters if the thing you care about is the extremes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
from odc.stac import load
from shapely.geometry import box, shape
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- Settings ----------------
BBOX = (55.10, 25.00, 55.40, 25.25)      # Dubai: city, desert, coast, parks
YEAR = "2024-01-01/2024-12-31"
MAX_CLOUD = 10
MATCH_WINDOW_DAYS = 8                     # how close the two scenes must be
CRS = "EPSG:32640"
RESOLUTION = 30                           # Landsat thermal native grid

BLOCKS = 6                                # spatial blocks, the Day 15 lesson
TRAIN_SAMPLES = 60000
SEED = 19

# Landsat Collection 2 Level-2 surface temperature scaling
ST_SCALE = 0.00341802
ST_OFFSET = 149.0

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
BAD_SCL = [0, 1, 3, 8, 9, 10]


def catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def kelvin_to_celsius(digital_number):
    """Landsat ST_B10 digital numbers to degrees Celsius.

    The scale and offset come from the collection metadata. Getting them
    wrong does not produce an error, it produces a temperature map that
    looks plausible and is off by tens of degrees, so the self-test
    checks the conversion against a known physical value.
    """
    kelvin = digital_number * ST_SCALE + ST_OFFSET
    return kelvin - 273.15


def decode_landsat_clouds(qa_pixel):
    """Bits 3 and 4 of QA_PIXEL are cloud and cloud shadow.

    Landsat packs its quality flags into single bits rather than the
    class codes Sentinel-2 uses, so this has to be done by masking.
    """
    cloud = (qa_pixel & (1 << 3)) > 0
    shadow = (qa_pixel & (1 << 4)) > 0
    dilated = (qa_pixel & (1 << 1)) > 0
    fill = (qa_pixel & 1) > 0
    return ~(cloud | shadow | dilated | fill)


def find_matched_pair(client):
    """A Landsat scene and a Sentinel-2 scene close together in time.

    Surface temperature changes completely from one day to the next, so
    predicting it from reflectance taken weeks away would be measuring
    nothing. The two acquisitions have to be near each other.
    """
    aoi = box(*BBOX)

    landsat = [it for it in client.search(
        collections=["landsat-c2-l2"], bbox=BBOX, datetime=YEAR,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD},
               "platform": {"in": ["landsat-8", "landsat-9"]}}).items()
        if "lwir11" in it.assets
        and shape(it.geometry).intersection(aoi).area / aoi.area >= 0.99]

    if not landsat:
        raise RuntimeError("No clear Landsat scene fully covers the box")

    sentinel = list(client.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=YEAR,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}}).items())

    s2_days = {}
    for it in sentinel:
        s2_days.setdefault(it.datetime.date(), []).append(it)

    covering_days = {}
    for day, items in s2_days.items():
        footprint = unary_union([shape(i.geometry) for i in items])
        if footprint.intersection(aoi).area / aoi.area >= 0.99:
            covering_days[day] = items

    if not covering_days:
        raise RuntimeError("No Sentinel-2 day fully covers the box")

    best = None
    for ls in landsat:
        ls_day = ls.datetime.date()
        for s2_day, items in covering_days.items():
            gap = abs((s2_day - ls_day).days)
            if gap > MATCH_WINDOW_DAYS:
                continue
            cloud = (ls.properties["eo:cloud_cover"]
                     + sum(i.properties["eo:cloud_cover"] for i in items) / len(items))
            score = (gap, cloud)
            if best is None or score < best[0]:
                best = (score, ls, s2_day, items)

    if best is None:
        raise RuntimeError(
            f"No Landsat and Sentinel-2 pair within {MATCH_WINDOW_DAYS} days. "
            f"Widen MATCH_WINDOW_DAYS or MAX_CLOUD.")

    (gap, cloud), ls, s2_day, s2_items = best
    print(f"  Landsat    {ls.datetime.date()}  "
          f"cloud {ls.properties['eo:cloud_cover']:.1f}%  ({ls.id})")
    print(f"  Sentinel-2 {s2_day}  "
          f"cloud {sum(i.properties['eo:cloud_cover'] for i in s2_items) / len(s2_items):.1f}%")
    print(f"  {gap} day(s) apart")
    return ls, s2_items


def load_pair(client, landsat_item, s2_items):
    """Surface temperature, and the reflectance used to predict it."""
    ls = load([landsat_item], bands=["lwir11", "qa_pixel"], bbox=BBOX,
              crs=CRS, resolution=RESOLUTION).isel(time=0)

    raw = ls["lwir11"].values.astype("float64")
    temperature = kelvin_to_celsius(raw)
    temperature[raw == 0] = np.nan
    temperature[~decode_landsat_clouds(ls["qa_pixel"].values)] = np.nan

    s2 = load(s2_items, bands=S2_BANDS + ["SCL"], bbox=BBOX, crs=CRS,
              resolution=RESOLUTION, groupby="solar_day").isel(time=0)

    offset = 1000 if any(
        float(i.properties.get("s2:processing_baseline", "0")) >= 4.0
        for i in s2_items) else 0

    clear = ~s2["SCL"].isin(BAD_SCL)
    bands = {}
    for b in S2_BANDS:
        arr = s2[b].values.astype("float32")
        nodata = arr == 0
        arr = (arr - offset) / 10000.0
        arr = np.clip(arr, 0, 1)
        arr[nodata | ~clear.values] = np.nan
        bands[b] = arr

    return temperature, bands


def build_features(bands):
    """Reflectance plus the indices that carry the physical signal."""
    def nd(a, b):
        denom = a + b
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denom > 0, (a - b) / denom, np.nan)

    features = dict(bands)
    features["NDVI"] = nd(bands["B08"], bands["B04"])      # vegetation cools
    features["NDBI"] = nd(bands["B11"], bands["B08"])      # built-up warms
    features["NDWI"] = nd(bands["B03"], bands["B08"])
    features["albedo"] = (0.356 * bands["B02"] + 0.130 * bands["B04"]
                          + 0.373 * bands["B08"] + 0.085 * bands["B11"]
                          + 0.072 * bands["B12"] - 0.0018) / 1.016
    return features


def block_index(shape_hw, blocks=BLOCKS):
    h, w = shape_hw
    row = (np.arange(h) * blocks // h)[:, None]
    col = (np.arange(w) * blocks // w)[None, :]
    return (row * blocks + col).astype(np.int16)


def report(name, truth, prediction):
    rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
    mae = float(mean_absolute_error(truth, prediction))
    r2 = float(r2_score(truth, prediction))
    bias = float((prediction - truth).mean())
    print(f"  {name:<28} R2 {r2:6.3f}   RMSE {rmse:5.2f} C   "
          f"MAE {mae:5.2f} C   bias {bias:+5.2f} C")
    return dict(name=name, r2=r2, rmse=rmse, mae=mae, bias=bias)


def self_test():
    """Check the conversion and the metrics against values that are fixed."""
    rng = np.random.default_rng(0)

    # The scale and offset must give physically sensible temperatures.
    # These reference points are computed from the published scaling
    # (DN * 0.00341802 + 149, in kelvin), not guessed, so the test fails
    # if the conversion is ever changed to something wrong.
    for dn, expected in ((44000, 26.24), (48000, 39.91), (50000, 46.75)):
        c = kelvin_to_celsius(dn)
        print(f"  ST digital number {dn} -> {c:5.1f} C "
              f"(expected {expected:.1f})")
        assert abs(c - expected) < 0.05, "the ST scaling is wrong"

    # And the conversion must be invertible, so a target temperature maps
    # back to a plausible digital number for a hot desert city
    dn_for_45 = (45 + 273.15 - ST_OFFSET) / ST_SCALE
    assert 40000 < dn_for_45 < 60000, "45 C maps outside the valid DN range"
    print(f"  45 C corresponds to digital number {dn_for_45:.0f}, "
          f"inside the valid range")

    # QA_PIXEL bit decoding
    clear_value = 0
    cloudy = 1 << 3
    assert decode_landsat_clouds(np.array([clear_value]))[0]
    assert not decode_landsat_clouds(np.array([cloudy]))[0]
    print("  QA_PIXEL bit 3 correctly flags cloud")

    # R-squared of predicting the mean is exactly zero, by construction
    truth = rng.normal(30, 5, 5000)
    assert abs(r2_score(truth, np.full_like(truth, truth.mean()))) < 1e-12
    assert abs(r2_score(truth, truth) - 1.0) < 1e-12
    print("  R2: exactly 0 for predicting the mean, 1 for a perfect fit")

    # R-squared depends on the SPREAD of the target, not only on the model.
    # Identical errors, different target variance, different R2. This is
    # the reason an R2 from one study cannot be compared with another's.
    error_sigma = 2.0
    for spread in (2.0, 5.0, 12.0):
        y = rng.normal(30, spread, 20000)
        p = y + rng.normal(0, error_sigma, 20000)
        r2 = r2_score(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))
        print(f"  target spread {spread:4.1f} C -> R2 {r2:5.3f} "
              f"with the SAME RMSE of {rmse:.2f} C")
    assert r2_score(rng.normal(30, 12, 20000),
                    rng.normal(30, 12, 20000) + rng.normal(0, 2, 20000)) > \
           r2_score(rng.normal(30, 2, 20000),
                    rng.normal(30, 2, 20000) + rng.normal(0, 2, 20000)), \
        "R2 should rise with target spread at fixed error"

    # Regression to the mean: a model fitted to minimise squared error
    # shrinks its predictions toward the middle, so the slope of
    # predicted against actual is below 1 even when the model is unbiased.
    x = rng.normal(0, 1, 20000)
    y = 3 * x + rng.normal(0, 3, 20000)
    fitted = LinearRegression().fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
    slope = LinearRegression().fit(fitted.reshape(-1, 1), y).coef_[0]
    spread_ratio = fitted.std() / y.std()
    print(f"  predictions are narrower than the truth: "
          f"spread ratio {spread_ratio:.3f}")
    assert spread_ratio < 1.0, "a least squares fit should shrink the spread"


def main():
    print("Self-test:")
    self_test()

    client = catalog()
    print("\nFinding a matched Landsat and Sentinel-2 pair...")
    landsat_item, s2_items = find_matched_pair(client)

    print("\nLoading...")
    temperature, bands = load_pair(client, landsat_item, s2_items)
    print(f"  scene: {temperature.shape[1]} x {temperature.shape[0]} px "
          f"at {RESOLUTION} m")

    features = build_features(bands)
    names = list(features)
    stack = np.dstack([features[n] for n in names])

    valid = np.isfinite(stack).all(axis=2) & np.isfinite(temperature)
    print(f"  usable pixels: {int(valid.sum()):,} "
          f"({100 * valid.mean():.1f}% of the scene)")

    X_all = stack[valid]
    y_all = temperature[valid]
    blocks_all = block_index(temperature.shape)[valid]

    print(f"\n  Surface temperature: {y_all.min():.1f} to {y_all.max():.1f} C, "
          f"mean {y_all.mean():.1f} C, spread {y_all.std():.2f} C")

    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X_all), min(TRAIN_SAMPLES * 2, len(X_all)),
                        replace=False)
    X, y, b = X_all[sample], y_all[sample], blocks_all[sample]

    held_blocks = rng.choice(BLOCKS * BLOCKS, BLOCKS * BLOCKS // 3,
                             replace=False)
    held = np.isin(b, held_blocks)
    print(f"  train {int((~held).sum()):,} px, test {int(held.sum()):,} px "
          f"({len(held_blocks)} of {BLOCKS * BLOCKS} spatial blocks held out)")

    # ---------- Baselines first ----------
    print(f"\n--- Baselines, on held-out blocks ---")
    results = []
    results.append(report("Predict the mean",
                          y[held], np.full(held.sum(), y[~held].mean())))

    ndvi_index = names.index("NDVI")
    ndvi_model = LinearRegression().fit(X[~held][:, [ndvi_index]], y[~held])
    results.append(report("NDVI alone, linear",
                          y[held], ndvi_model.predict(X[held][:, [ndvi_index]])))
    print(f"    NDVI slope: {ndvi_model.coef_[0]:+.2f} C per unit NDVI "
          f"(here the NDVI range runs water to bare sand, so higher NDVI is hotter)")

    linear = LinearRegression().fit(X[~held], y[~held])
    results.append(report("All features, linear",
                          y[held], linear.predict(X[held])))

    forest = RandomForestRegressor(n_estimators=200, random_state=SEED,
                                   n_jobs=-1, min_samples_leaf=3)
    forest.fit(X[~held], y[~held])
    prediction = forest.predict(X[held])
    results.append(report("All features, random forest", y[held], prediction))

    # ---------- What R-squared hides ----------
    print(f"\n--- What the R-squared does not say ---")
    truth = y[held]
    spread_ratio = prediction.std() / truth.std()
    slope = LinearRegression().fit(
        prediction.reshape(-1, 1), truth).coef_[0]

    print(f"  Target spread on the test blocks: {truth.std():.2f} C")
    print(f"  Prediction spread:                {prediction.std():.2f} C")
    if spread_ratio < 0.98:
        print(f"  Ratio {spread_ratio:.3f}: the model's range is "
              f"{100 * (1 - spread_ratio):.0f}% narrower than reality.")
    else:
        print(f"  Ratio {spread_ratio:.3f}: the spread is reproduced, so "
              f"there is little shrinkage here.")
    print(f"  Regressing truth on prediction gives a slope of {slope:.2f} "
          f"(1.00 would be unbiased at every level).")

    coldest = truth < np.percentile(truth, 10)
    hottest = truth > np.percentile(truth, 90)
    cold_bias = float((prediction[coldest] - truth[coldest]).mean())
    hot_bias = float((prediction[hottest] - truth[hottest]).mean())

    print(f"\n  Overall bias:                      "
          f"{float((prediction - truth).mean()):+.2f} C")
    print(f"  Bias in the coldest 10% of pixels: {cold_bias:+.2f} C")
    print(f"  Bias in the hottest 10% of pixels: {hot_bias:+.2f} C")

    if cold_bias > 0.1 and hot_bias < -0.1:
        print(f"\n  Those two have OPPOSITE signs, which is the signature of "
              f"a model fitted\n  to minimise squared error: it pulls "
              f"predictions toward the middle. The\n  overall bias of "
              f"{float((prediction - truth).mean()):+.2f} C hides it entirely, "
              f"because the two errors cancel\n  in the average. An urban "
              f"heat study looks at exactly the extremes this\n  gets wrong, "
              f"by {abs(cold_bias):.2f} C and {abs(hot_bias):.2f} C in "
              f"opposite directions.")
    else:
        print(f"\n  The extremes are not biased in opposite directions here, "
              f"so this model is\n  not showing the usual pull toward the "
              f"middle.")

    importance = sorted(zip(names, forest.feature_importances_),
                        key=lambda p: -p[1])
    print(f"\n  Most useful predictors: " +
          ", ".join(f"{n} ({100 * v:.0f}%)" for n, v in importance[:4]))

    # ---------- Full map ----------
    predicted_map = np.full(temperature.shape, np.nan)
    predicted_map[valid] = forest.predict(X_all)
    residual_map = predicted_map - temperature

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    lo, hi = np.nanpercentile(temperature, [2, 98])

    im = axes[0, 0].imshow(temperature, cmap="inferno", vmin=lo, vmax=hi)
    axes[0, 0].set_title(f"Landsat surface temperature "
                         f"({landsat_item.datetime.date()})")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, label="deg C")

    im = axes[0, 1].imshow(predicted_map, cmap="inferno", vmin=lo, vmax=hi)
    axes[0, 1].set_title("Predicted from Sentinel-2 reflectance")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, label="deg C")

    limit = np.nanpercentile(np.abs(residual_map), 98)
    im = axes[0, 2].imshow(residual_map, cmap="bwr", vmin=-limit, vmax=limit)
    axes[0, 2].set_title("Residual (predicted minus actual)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, label="deg C")

    ax = axes[1, 0]
    show = rng.choice(len(truth), min(8000, len(truth)), replace=False)
    ax.scatter(truth[show], prediction[show], s=2, alpha=0.2, color="tab:blue")
    span = [truth.min(), truth.max()]
    ax.plot(span, span, "k--", linewidth=1.4, label="Perfect (slope 1)")
    fit = LinearRegression().fit(truth.reshape(-1, 1), prediction)
    ax.plot(span, fit.predict(np.array(span).reshape(-1, 1)),
            color="tab:red", linewidth=1.8,
            label=f"Actual fit (slope {fit.coef_[0]:.2f})")
    ax.set_xlabel("Measured temperature (C)")
    ax.set_ylabel("Predicted (C)")
    ax.set_title(f"Predicted vs measured (slope {fit.coef_[0]:.2f}; "
                 f"below 1 means pulled to the middle)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ndvi_values = features["NDVI"][valid]
    show2 = rng.choice(len(ndvi_values), min(8000, len(ndvi_values)),
                       replace=False)
    ax.scatter(ndvi_values[show2], y_all[show2], s=2, alpha=0.2,
               color="tab:green")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Surface temperature (C)")
    ax.set_title(f"The urban heat island signal "
                 f"({ndvi_model.coef_[0]:+.1f} C per NDVI unit)", fontsize=10)
    ax.grid(alpha=0.3)

    # Mark where the model is biased, which the overall figure hides
    ax.axhline(np.percentile(truth, 10), color="tab:blue", linestyle=":",
               linewidth=1, label=f"Coldest 10%: {cold_bias:+.2f} C bias")
    ax.axhline(np.percentile(truth, 90), color="tab:red", linestyle=":",
               linewidth=1, label=f"Hottest 10%: {hot_bias:+.2f} C bias")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    labels = [r["name"].replace(", ", "\n") for r in results]
    ax.bar(range(len(results)), [r["rmse"] for r in results],
           color=["tab:gray", "tab:olive", "tab:orange", "tab:blue"])
    ax.set_xticks(range(len(results)), labels, fontsize=7)
    ax.set_ylabel("RMSE (deg C)")
    ax.set_title("RMSE is in real units. R-squared is not.")
    ax.grid(alpha=0.3, axis="y")
    for i, r in enumerate(results):
        ax.text(i, r["rmse"] + 0.05, f"{r['rmse']:.2f}\nR2 {r['r2']:.2f}",
                ha="center", fontsize=7)

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2]):
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Day 19: Predicting land surface temperature, "
                 "and why R-squared flatters", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "lst_regression.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
