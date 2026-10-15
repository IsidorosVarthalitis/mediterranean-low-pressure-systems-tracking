

# -*- coding: utf-8 -*-
"""Top-N Cyclone Selection with Non-Parametric Bootstrap and Permutation Testing.

Author: Isidoros Varthalitis
Description:
    Algorithm for spatiotemporal coupling between ERA5 reanalysis precipitation
    and SyCLoPS cyclone track database over the Eastern Mediterranean / Crete,
    incorporating moisture flux divergence (VIMDF) optimization and trend analysis.
"""

import os
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# ==============================================================================
# Configuration Parameters
# ==============================================================================
DATA_DIR = Path("data")
SYCLOPS_PATH = Path("data/SyCLoPS_classified_ERA5_1940_2024.parquet")

REGION_TO_TEST = "all"
THRESHOLD_RAIN = 5  # Options: 1, 3, 5 mm/h
TOP_N = 1          # Options: 1 (Top-1) or 3 (Top-3)

PERIOD_COLORS = {
    "1960_1975": "#1f77b4",
    "1976_1991": "#2ca02c",
    "1992_2007": "#ff7f0e",
    "2008_2024": "#d62728",
}

DOT_SIZE = 12 if THRESHOLD_RAIN >= 3 else 5
DOT_ALPHA = 0.8 if THRESHOLD_RAIN >= 3 else 0.4
KM_PER_DEG = 111.12


# ==============================================================================
# Storm Extraction & Hybrid Coupling Function
# ==============================================================================
def get_storm_centroids(period_label, sub_region="all", top_n=1):
    start_y, end_y = map(int, period_label.split("_"))
    all_files = [
        str(DATA_DIR / f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".nc")
        and "rain" in f
        and any(str(y) in f for y in range(start_y, end_y + 1))
    ]

    if not all_files:
        print(f"No precipitation files found for period: {period_label}")
        return pd.DataFrame()

    rainy_times = []
    for file_path in all_files:
        with xr.open_dataset(file_path) as ds:
            t_n = "valid_time" if "valid_time" in ds.coords else "time"
            lat_n = "latitude" if "latitude" in ds.coords else "lat"
            lon_n = "longitude" if "longitude" in ds.coords else "lon"

            if sub_region == "west":
                l_lims = [23.5, 25.0]
            elif sub_region == "east":
                l_lims = [25.0, 26.5]
            else:
                l_lims = [23.5, 26.5]

            is_desc = ds[lat_n].values[0] > ds[lat_n].values[-1]
            lat_slice = slice(36.0, 34.5) if is_desc else slice(34.5, 36.0)

            ds_sub = ds.sel({lat_n: lat_slice, lon_n: slice(l_lims[0], l_lims[1])})
            tp_val = ds_sub.tp * 1000

            if tp_val.size == 0:
                continue

            max_rain = tp_val.max(dim=[lat_n, lon_n]).compute()
            times = max_rain[t_n].where(max_rain > THRESHOLD_RAIN, drop=True).values
            rainy_times.extend(times)

    if not rainy_times:
        print(f"No rainfall events exceeding {THRESHOLD_RAIN} mm/h for: {period_label}")
        return pd.DataFrame()

    df_p = pd.read_parquet(
        SYCLOPS_PATH,
        columns=["LAT", "LON", "ISOTIME"],
        filters=[
            ("ISOTIME", ">=", pd.Timestamp(f"{start_y}-01-01")),
            ("ISOTIME", "<=", pd.Timestamp(f"{end_y}-12-31")),
        ],
    )

    df_rain = pd.DataFrame({"ISOTIME": pd.to_datetime(rainy_times)})
    df_rain["ISOTIME"] = df_rain["ISOTIME"].dt.round("3h")

    df_c = pd.merge(df_rain, df_p, on="ISOTIME", how="inner")
    df_c = df_c[(df_c["LAT"].between(28, 42)) & (df_c["LON"].between(18, 32))]

    if df_c.empty:
        print(f"No cyclonic tracks within bounds for: {period_label}")
        return pd.DataFrame()

    final_responsible_cyclones = []

    for isotime, group in df_c.groupby("ISOTIME"):
        year = int(isotime.year)
        mfd_file_path = DATA_DIR / f"era5_mfd_{year}.nc"

        group_c = group.copy().reset_index(drop=True)
        scores = []

        try:
            with xr.open_dataset(mfd_file_path) as ds_mfd:
                t_n = "valid_time" if "valid_time" in ds_mfd.coords else "time"
                lat_n = "latitude" if "latitude" in ds_mfd.coords else "lat"
                lon_n = "longitude" if "longitude" in ds_mfd.coords else "lon"

                mfd_var_name = "vimdf" if "vimdf" in ds_mfd else "mfd"
                is_lat_desc = ds_mfd[lat_n].values[0] > ds_mfd[lat_n].values[-1]

                for _, row in group_c.iterrows():
                    c_lat, c_lon = row["LAT"], row["LON"]
                    dist = np.sqrt((c_lat - 35.2) ** 2 + (c_lon - 25.0) ** 2)
                    dist = max(dist, 0.1)

                    lat_slice = (
                        slice(c_lat + 2.5, c_lat - 2.5)
                        if is_lat_desc
                        else slice(c_lat - 2.5, c_lat + 2.5)
                    )

                    try:
                        mfd_val = (
                            ds_mfd[mfd_var_name]
                            .sel({
                                t_n: isotime,
                                lat_n: lat_slice,
                                lon_n: slice(c_lon - 2.5, c_lon + 2.5),
                            })
                            .mean()
                            .compute()
                            .item()
                        )
                    except Exception:
                        mfd_val = 0.0

                    convergence = -mfd_val
                    hybrid_score = dist / (1.0 + max(0.0, convergence * 1000.0))
                    scores.append(hybrid_score)

            group_c["score"] = scores
            group_c = group_c.sort_values("score", ascending=True)
            final_responsible_cyclones.append(group_c.head(top_n))

        except Exception:
            group_c["dist"] = np.sqrt(
                (group_c["LAT"] - 35.2) ** 2 + (group_c["LON"] - 25.0) ** 2
            )
            group_c = group_c.sort_values("dist", ascending=True)
            final_responsible_cyclones.append(group_c.head(top_n))

    if final_responsible_cyclones:
        df_final_period = pd.concat(final_responsible_cyclones, ignore_index=True)
        return df_final_period[["LAT", "LON", "ISOTIME"]]
    return pd.DataFrame()


# ==============================================================================
# Period Aggregation and Standard Error Calculations
# ==============================================================================
final_stats = []
all_data = {}

for p in PERIOD_COLORS.keys():
    print(f"Processing period: {p} (Top-{TOP_N})...")
    df = get_storm_centroids(p, sub_region=REGION_TO_TEST, top_n=TOP_N)
    if not df.empty:
        all_data[p] = df
        lats = df["LAT"].values
        n_events = len(lats)

        mean_lat = float(np.mean(lats))
        std_lat = float(np.std(lats, ddof=1))
        se_lat = std_lat / np.sqrt(n_events)

        final_stats.append({
            "Period": p,
            "Lat": mean_lat,
            "Lon": float(df["LON"].mean()),
            "N": n_events,
            "Std_Lat": std_lat,
            "SE_Lat": se_lat,
            "SE_km": se_lat * KM_PER_DEG,
        })

df_final = pd.DataFrame(final_stats)


# ==============================================================================
# Non-Parametric Bootstrap & Permutation Testing
# ==============================================================================
pval_mod = np.nan
pval_full = np.nan
p_val_permutation = np.nan

if not df_final.empty and len(df_final) == 4:
    print(f"\nRunning Bootstrap analysis (2,000 iterations) for Top-{TOP_N}...")

    np.random.seed(42)
    n_boot = 2000
    alpha = 0.05

    periods_order = list(PERIOD_COLORS.keys())
    x_full = np.arange(4)
    x_mod = np.arange(1, 4)

    boot_shifts_full = np.empty(n_boot)
    boot_shifts_mod = np.empty(n_boot)
    boot_lines_mod = np.empty((n_boot, 3))

    for b in range(n_boot):
        sampled_means = [
            np.mean(
                np.random.choice(
                    all_data[p]["LAT"].values, size=len(all_data[p]), replace=True
                )
            )
            for p in periods_order
        ]
        y_b = np.array(sampled_means)

        # Full Trend (1960-2024)
        p_f = np.polyfit(x_full, y_b, deg=1)
        boot_shifts_full[b] = (p_f[0] * x_full[-1] + p_f[1]) - (
            p_f[0] * x_full[0] + p_f[1]
        )

        # Modern Trend (1976-2024)
        p_m = np.polyfit(x_mod, y_b[1:], deg=1)
        boot_shifts_mod[b] = (p_m[0] * x_mod[-1] + p_m[1]) - (
            p_m[0] * x_mod[0] + p_m[1]
        )
        boot_lines_mod[b, :] = np.poly1d(p_m)(x_mod)

    low_pct = 100 * (alpha / 2.0)
    high_pct = 100 * (1.0 - alpha / 2.0)

    ci_mod_deg = (
        np.percentile(boot_shifts_mod, low_pct),
        np.percentile(boot_shifts_mod, high_pct),
    )
    ci_mod_km = (ci_mod_deg[0] * KM_PER_DEG, ci_mod_deg[1] * KM_PER_DEG)
    se_mod_km = np.std(boot_shifts_mod, ddof=1) * KM_PER_DEG
    pval_mod = np.mean(boot_shifts_mod <= 0.0)

    ci_full_deg = (
        np.percentile(boot_shifts_full, low_pct),
        np.percentile(boot_shifts_full, high_pct),
    )
    ci_full_km = (ci_full_deg[0] * KM_PER_DEG, ci_full_deg[1] * KM_PER_DEG)
    se_full_km = np.std(boot_shifts_full, ddof=1) * KM_PER_DEG
    pval_full = np.mean(boot_shifts_full <= 0.0)

    line_mod_lower = np.percentile(boot_lines_mod, low_pct, axis=0)
    line_mod_upper = np.percentile(boot_lines_mod, high_pct, axis=0)

    # Permutation Test (Null Hypothesis Verification)
    print("Executing permutation test (1,000 resamples)...")
    all_lats_combined = np.concatenate(
        [all_data[p]["LAT"].values for p in periods_order[1:]]
    )
    group_sizes = [len(all_data[p]) for p in periods_order[1:]]
    null_shifts = []

    for _ in range(1000):
        shuffled = np.random.permutation(all_lats_combined)
        split_1 = group_sizes[0]
        split_2 = group_sizes[0] + group_sizes[1]

        mean_1 = np.mean(shuffled[:split_1])
        mean_2 = np.mean(shuffled[split_1:split_2])
        mean_3 = np.mean(shuffled[split_2:])

        p_null = np.polyfit(x_mod, [mean_1, mean_2, mean_3], deg=1)
        shift_null = (p_null[0] * x_mod[-1] + p_null[1]) - (
            p_null[0] * x_mod[0] + p_null[1]
        )
        null_shifts.append(shift_null)

    observed_mod_shift = (
        np.polyfit(x_mod, df_final["Lat"].values[1:], 1)[0]
        * (x_mod[-1] - x_mod[0])
    )
    p_val_permutation = np.mean(np.array(null_shifts) >= observed_mod_shift)


# ==============================================================================
# Visualization
# ==============================================================================
if not df_final.empty:
    fig = plt.figure(figsize=(24, 14))
    plt.suptitle(
        f"Storm Track Analysis: {REGION_TO_TEST.upper()} Crete "
        f"(Threshold: {THRESHOLD_RAIN} mm/h | Top-{TOP_N})",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )

    for i, (p, col) in enumerate(PERIOD_COLORS.items()):
        ax = fig.add_subplot(2, 4, i + 1, projection=ccrs.PlateCarree())
        ax.set_extent([18, 32, 28, 42])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        if p in all_data:
            df_p = all_data[p]
            ax.scatter(
                df_p["LON"],
                df_p["LAT"],
                color=col,
                s=DOT_SIZE,
                alpha=DOT_ALPHA,
                edgecolors="none",
                zorder=5,
            )
            ax.set_title(f"{p}\nN={len(df_p)}", fontsize=12, fontweight="bold")

    ax_map = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    lat_m, lon_m = df_final["Lat"].mean(), df_final["Lon"].mean()
    ax_map.set_extent([lon_m - 1.2, lon_m + 1.2, lat_m - 0.8, lat_m + 0.8])
    ax_map.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax_map.add_feature(cfeature.LAND, facecolor="#fdfdfd")

    for i, row in df_final.iterrows():
        ax_map.plot(
            row["Lon"],
            row["Lat"],
            "o",
            color=PERIOD_COLORS[row["Period"]],
            markersize=16,
            markeredgecolor="k",
            zorder=10,
            label=row["Period"],
        )

        if row["Period"] == "1976_1991":
            x_text_offset = -0.32
            y_text_offset = -0.04
        else:
            x_text_offset = 0.03
            y_text_offset = 0.03

        ax_map.text(
            row["Lon"] + x_text_offset,
            row["Lat"] + y_text_offset,
            f"{row['Lat']:.4f}°N\n±{row['SE_km']:.1f}km",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax_map.set_title(
        "B. Focused Spatial Shift (Zoomed View)", fontsize=16, fontweight="bold"
    )
    ax_map.legend(loc="lower right", fontsize=10)

    gl = ax_map.gridlines(draw_labels=True, linestyle="--", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False

    ax_trend = fig.add_subplot(2, 2, 4)
    x, y = np.arange(len(df_final)), df_final["Lat"].values

    z_f = np.polyfit(x, y, 1)
    z_m = np.polyfit(x[1:], y[1:], 1)

    total_shift_full = np.poly1d(z_f)(x[-1]) - np.poly1d(z_f)(x[0])
    total_shift_modern = np.poly1d(z_m)(x[-1]) - np.poly1d(z_m)(x[1])

    ax_trend.plot(
        x,
        np.poly1d(z_f)(x),
        "k--",
        alpha=0.4,
        label=(
            f"Full Trend (1960-2024): {total_shift_full:+.3f}°\n"
            f"(Boot p={pval_full:.4f})"
        ),
    )

    ax_trend.plot(
        x[1:],
        np.poly1d(z_m)(x[1:]),
        color="red",
        linewidth=3,
        label=(
            f"Modern Trend (Post-1976): {total_shift_modern:+.3f}°\n"
            f"(Boot p={pval_mod:.4f} | Perm p={p_val_permutation:.4f})"
        ),
    )

    if len(df_final) == 4:
        ax_trend.fill_between(
            x[1:],
            line_mod_lower,
            line_mod_upper,
            color="red",
            alpha=0.15,
            label=f"Modern Trend 95% CI (Bootstrap Top-{TOP_N})",
        )

    for i, row in df_final.iterrows():
        ax_trend.errorbar(
            i,
            row["Lat"],
            yerr=row["SE_Lat"],
            fmt="s",
            color=PERIOD_COLORS[row["Period"]],
            markersize=12,
            markeredgecolor="k",
            capsize=8,
            capthick=1.5,
            elinewidth=1.5,
            zorder=10,
        )

    ax_trend.set_title(
        f"C. Latitudinal Trend Analysis (Top-{TOP_N})",
        fontsize=16,
        fontweight="bold",
    )
    ax_trend.set_xticks(x)
    ax_trend.set_xticklabels(df_final["Period"])
    ax_trend.set_ylabel("Mean Latitude (°N)", fontsize=12)
    ax_trend.grid(True, linestyle=":", alpha=0.6)
    ax_trend.legend(fontsize=10, loc="upper left")

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
    plt.show()

    # ==============================================================================
    # Summary Output
    # ==============================================================================
    print("\n" + "=" * 80)
    print(f"Summary Statistics: Centroids & SE [Threshold: {THRESHOLD_RAIN} mm/h | Top-{TOP_N}]")
    print("=" * 80)
    print(
        df_final[[
            "Period",
            "N",
            "Lat",
            "Lon",
            "Std_Lat",
            "SE_Lat",
            "SE_km",
        ]].to_string(index=False)
    )

    if len(df_final) == 4:
        print("\n" + "=" * 80)
        print(f"Trend Statistical Significance (Bootstrap 2,000 runs | Top-{TOP_N})")
        print("=" * 80)
        print("MODERN SHIFT (1976-2024):")
        print(f"  - Point Estimate: {total_shift_modern:+.4f}° ({total_shift_modern*KM_PER_DEG:+.2f} km)")
        print(f"  - 95% Confidence Interval: [{ci_mod_deg[0]:+.4f}°, {ci_mod_deg[1]:+.4f}°] -> [{ci_mod_km[0]:+.2f} km, {ci_mod_km[1]:+.2f} km]")
        print(f"  - Standard Error (SE): ±{se_mod_km:.2f} km")
        print(f"  - Bootstrap p-value (H0: Shift <= 0): p = {pval_mod:.4f}")
        print(f"  - Permutation Null Test p-value: p = {p_val_permutation:.4f}")

        print("\nFULL SHIFT (1960-2024):")
        print(f"  - Point Estimate: {total_shift_full:+.4f}° ({total_shift_full*KM_PER_DEG:+.2f} km)")
        print(f"  - 95% Confidence Interval: [{ci_full_deg[0]:+.4f}°, {ci_full_deg[1]:+.4f}°] -> [{ci_full_km[0]:+.2f} km, {ci_full_km[1]:+.2f} km]")
        print(f"  - Standard Error (SE): ±{se_full_km:.2f} km")
        print(f"  - Bootstrap p-value (H0: Shift <= 0): p = {pval_full:.4f}")
        print("=" * 80)