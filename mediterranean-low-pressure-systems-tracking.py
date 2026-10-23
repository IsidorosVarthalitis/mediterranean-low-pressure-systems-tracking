# -*- coding: utf-8 -*-
"""Created on Wed Oct  7 15:50:47 2026.

@author: Isidoros Varthalitis

Universal script for Top-N Low-pressure Systems Selection with Block Bootstrap &
Cluster-Robust Uncertainty Analysis based on SyCLoPS Track ID (TID).
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
# 1. ΠΑΡΑΜΕΤΡΟΙ ΕΚΤΕΛΕΣΗΣ
# ==============================================================================
# Τα δεδομένα αναμένονται στον φάκελο data/ (βλ. README).
# Για τοπική εκτέλεση αλλάξτε τις διαδρομές αν χρειάζεται.
data_dir = Path("data")
syclops_path = data_dir / "SyCLoPS_classified_ERA5_1940_2024.parquet"

REGION_TO_TEST = "all"
THRESHOLD_RAIN = 5  # Επιλογές: 1, 3, 5
TOP_N = 1  # Επιλογές: 1 (Top-1) ή 3 (Top-3)

colors = {
    "1960_1975": "#1f77b4",
    "1976_1991": "#2ca02c",
    "1992_2007": "#ff7f0e",
    "2008_2024": "#d62728",
}

dot_size = 12 if THRESHOLD_RAIN >= 3 else 5
dot_alpha = 0.8 if THRESHOLD_RAIN >= 3 else 0.4
KM_PER_DEG = 111.12

# Χρονική διάρκεια μεταξύ των μέσων των περιόδων (σε δεκαετίες)
DECADES_FULL = (2016.0 - 1967.5) / 10.0  # 4.85 δεκαετίες
DECADES_MODERN = (2016.0 - 1983.5) / 10.0  # 3.25 δεκαετίες
N_BOOT = 2000


def fmt_p(p, digits=4):
  """p-value bootstrap: αν δεν βγει κανένα δείγμα ≤ 0 γράφεται p < 1/N_BOOT."""
  if not np.isfinite(p):
    return "n/a"
  if p == 0.0:
    return f"< {1.0 / N_BOOT:.4f}"
  return f"= {p:.{digits}f}"


# ==============================================================================
# 2. ΣΥΝΑΡΤΗΣΗ ΑΝΑΚΤΗΣΗΣ ΚΥΚΛΩΝΩΝ (TOP-N SELECTION ΜΕ TID)
# ==============================================================================
def get_storm_centroids(period_label, sub_region="all", top_n=1):
  local_fallback = 0  # υποψήφια κέντρα χωρίς MFD
  local_fallback_events = 0  # χρονικά βήματα (γεγονότα) με ≥1 κέντρο χωρίς MFD
  start_y, end_y = map(int, period_label.split("_"))
  all_files = [
      str(data_dir / f)
      for f in os.listdir(data_dir)
      if f.endswith(".nc")
      and "rain" in f
      and any(str(y) in f for y in range(start_y, end_y + 1))
  ]

  if not all_files:
    print(f"Δεν βρέθηκαν αρχεία βροχής για: {period_label}")
    return pd.DataFrame(), 0, 0

  rainy_times = []
  for file_path in all_files:
    with xr.open_dataset(file_path) as ds:
      t_n = "valid_time" if "valid_time" in ds.coords else "time"
      lat_n = "latitude" if "latitude" in ds.coords else "lat"
      lon_n = "longitude" if "longitude" in ds.coords else "lon"

      l_lims = (
          [23.5, 25.0]
          if sub_region == "west"
          else [25.0, 26.5]
          if sub_region == "east"
          else [23.5, 26.5]
      )

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
    print(
        f"Δεν βρέθηκαν υετικά γεγονότα > {THRESHOLD_RAIN} mm/h για:"
        f" {period_label}"
    )
    return pd.DataFrame(), 0, 0

  # Διαβάζουμε και τη στήλη TID
  df_p = pd.read_parquet(
      syclops_path,
      columns=["LAT", "LON", "ISOTIME", "TID"],
      filters=[
          ("ISOTIME", ">=", pd.Timestamp(f"{start_y}-01-01")),
          ("ISOTIME", "<", pd.Timestamp(f"{end_y + 1}-01-01")),
      ],
  )

  df_rain = pd.DataFrame({"ISOTIME": pd.to_datetime(rainy_times)})
  df_rain["ISOTIME"] = df_rain["ISOTIME"].dt.round("3h")
  df_rain = df_rain.drop_duplicates(subset=["ISOTIME"]).reset_index(drop=True)

  df_c = pd.merge(df_rain, df_p, on="ISOTIME", how="inner")
  df_c = df_c[(df_c["LAT"].between(28, 42)) & (df_c["LON"].between(18, 32))]

  if df_c.empty:
    print(f"Κανένας κυκλώνας εντός ορίων για: {period_label}")
    return pd.DataFrame(), 0, 0

  final_responsible_cyclones = []
  cos_lat_crete = np.cos(np.radians(35.2))

  def distance_only(group_c):
    """Fallback κατάταξη μόνο με απόσταση (όταν δεν υπάρχει αρχείο MFD)."""
    d_lat = group_c["LAT"] - 35.2
    d_lon = (group_c["LON"] - 25.0) * cos_lat_crete
    group_c["dist"] = np.sqrt(d_lat**2 + d_lon**2)
    return group_c.sort_values("dist", ascending=True).head(top_n)

  # Το αρχείο MFD κάθε έτους ανοίγει μία φορά (όχι ανά 3ωρο βήμα).
  for year, df_year in df_c.groupby(df_c["ISOTIME"].dt.year):
    mfd_file_path = data_dir / f"era5_mfd_{int(year)}.nc"
    try:
      ds_mfd = xr.open_dataset(mfd_file_path)
      t_n = "valid_time" if "valid_time" in ds_mfd.coords else "time"
      lat_n = "latitude" if "latitude" in ds_mfd.coords else "lat"
      lon_n = "longitude" if "longitude" in ds_mfd.coords else "lon"
      mfd_var_name = "vimdf" if "vimdf" in ds_mfd else "mfd"
      mfd_da = ds_mfd[mfd_var_name]
      is_lat_desc = ds_mfd[lat_n].values[0] > ds_mfd[lat_n].values[-1]
    except Exception:
      ds_mfd = None

    if ds_mfd is None:
      for isotime, group in df_year.groupby("ISOTIME"):
        group_c = group.copy().reset_index(drop=True)
        local_fallback += len(group_c)
        local_fallback_events += 1
        final_responsible_cyclones.append(distance_only(group_c))
      continue

    with ds_mfd:
      for isotime, group in df_year.groupby("ISOTIME"):
        group_c = group.copy().reset_index(drop=True)
        scores = []
        event_fallback = False

        for _, row in group_c.iterrows():
          c_lat, c_lon = row["LAT"], row["LON"]

          d_lat = c_lat - 35.2
          d_lon = (c_lon - 25.0) * cos_lat_crete
          dist = np.sqrt(d_lat**2 + d_lon**2)
          dist = max(dist, 0.1)

          lat_slice = (
              slice(c_lat + 2.5, c_lat - 2.5)
              if is_lat_desc
              else slice(c_lat - 2.5, c_lat + 2.5)
          )

          try:
            mfd_val = (
                mfd_da.sel({
                    t_n: isotime,
                    lat_n: lat_slice,
                    lon_n: slice(c_lon - 2.5, c_lon + 2.5),
                })
                .mean()
                .compute()
                .item()
            )
          except Exception:
            mfd_val = np.nan

          # Κενό παράθυρο MFD (εκτός ορίων αρχείου) δίνει NaN χωρίς σφάλμα:
          # μετριέται κι αυτό ως fallback.
          if not np.isfinite(mfd_val):
            mfd_val = 0.0
            local_fallback += 1
            event_fallback = True

          convergence = -mfd_val
          hybrid_score = dist / (1.0 + max(0.0, convergence * 1000.0))
          scores.append(hybrid_score)

        local_fallback_events += int(event_fallback)
        group_c["score"] = scores
        group_c = group_c.sort_values("score", ascending=True)
        final_responsible_cyclones.append(group_c.head(top_n))

  if final_responsible_cyclones:
    df_final_period = pd.concat(final_responsible_cyclones, ignore_index=True)
    return (
        df_final_period[["LAT", "LON", "ISOTIME", "TID"]].copy(),
        local_fallback,
        local_fallback_events,
    )
  return pd.DataFrame(), 0, 0


# ==============================================================================
# 3. CLUSTER-ROBUST SE ΥΠΟΛΟΓΙΣΜΕΝΟ ΑΝΑ TRACK ID (TID)
# ==============================================================================
def cluster_se(df):
  """Τυπικό σφάλμα του μέσου πλάτους με ομαδοποίηση ανά SyCLoPS TID."""
  lats = df["LAT"].values
  m = lats.mean()
  n = len(lats)
  resid = (df["LAT"] - m).groupby(df["event"]).sum().values
  g = len(resid)
  if g < 2:
    return np.nan
  var = (g / (g - 1)) * np.sum(resid**2) / (n**2)
  return float(np.sqrt(var))


# ==============================================================================
# 4. ΕΠΕΞΕΡΓΑΣΙΑ & ΥΠΟΛΟΓΙΣΜΟΣ SE
# ==============================================================================
final_stats = []
all_data = {}
total_mfd_fallbacks = 0
total_mfd_fallback_events = 0

for p in colors.keys():
  print(f"Επεξεργασία περιόδου: {p} (Top-{TOP_N} με SyCLoPS TID)...")
  df, fb_count, fb_events = get_storm_centroids(
      p, sub_region=REGION_TO_TEST, top_n=TOP_N
  )
  total_mfd_fallbacks += fb_count
  total_mfd_fallback_events += fb_events

  if not df.empty:
    # Ορίζουμε το event βάσει του TID της SyCLoPS
    # Σημεία χωρίς TID γίνονται ξεχωριστά blocks (αλλιώς το groupby τα πετάει)
    df["event"] = df["TID"].astype(object)
    na = df["event"].isna()
    df.loc[na, "event"] = [f"noTID_{p}_{i}" for i in range(int(na.sum()))]
    df["event"] = df["event"].astype(str)
    print(f"   TID: NaN={int(na.sum())}, tracks={df['event'].nunique()},"
          f" steps={len(df)}")
    all_data[p] = df
    lats = df["LAT"].values
    n_timesteps = len(lats)

    mean_lat = float(np.mean(lats))
    std_lat = float(np.std(lats, ddof=1))
    se_lat = cluster_se(df)

    final_stats.append({
        "Period": p,
        "Lat": mean_lat,
        "Lon": float(df["LON"].mean()),
        "N_timesteps": n_timesteps,
        "N_tracks": int(df["event"].nunique()),
        "Std_Lat": std_lat,
        "SE_Lat": se_lat,
        "SE_km": se_lat * KM_PER_DEG,
    })

df_final = pd.DataFrame(final_stats)


# ==============================================================================
# 5. BLOCK BOOTSTRAPPING ΑΝΑ SYCLOPS TRACK
# ==============================================================================
pval_mod_one = np.nan
pval_mod_two = np.nan
pval_full_one = np.nan
pval_full_two = np.nan

if not df_final.empty and len(df_final) == 4:
  print(
      "\nΕκτέλεση Block Bootstrapping (2.000 επαναλήψεις ανά Track ID)"
      f" για Top-{TOP_N}..."
  )

  np.random.seed(42)
  n_boot = N_BOOT
  alpha = 0.05

  periods_order = list(colors.keys())
  x_full = np.arange(4)
  x_mod = np.arange(1, 4)

  # Κάθε μπλοκ = όλες οι θέσεις που αντιστοιχούν στο ίδιο Track ID (TID)
  blocks = {
      p: [g["LAT"].values for _, g in all_data[p].groupby("event")]
      for p in periods_order
  }

  boot_shifts_full = np.empty(n_boot)
  boot_shifts_mod = np.empty(n_boot)
  boot_lines_mod = np.empty((n_boot, 3))

  for b in range(n_boot):
    sampled_means = []
    for p in periods_order:
      k = len(blocks[p])
      idx = np.random.randint(0, k, k)
      sampled_means.append(np.concatenate([blocks[p][i] for i in idx]).mean())
    y_b = np.array(sampled_means)

    # 1. Full Trend (1960-2024)
    p_f = np.polyfit(x_full, y_b, deg=1)
    boot_shifts_full[b] = (p_f[0] * x_full[-1] + p_f[1]) - (
        p_f[0] * x_full[0] + p_f[1]
    )

    # 2. Modern Trend (1976-2024)
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

  pval_mod_one = np.mean(boot_shifts_mod <= 0.0)
  pval_mod_two = 2.0 * min(pval_mod_one, 1.0 - pval_mod_one)

  ci_full_deg = (
      np.percentile(boot_shifts_full, low_pct),
      np.percentile(boot_shifts_full, high_pct),
  )
  ci_full_km = (ci_full_deg[0] * KM_PER_DEG, ci_full_deg[1] * KM_PER_DEG)
  se_full_km = np.std(boot_shifts_full, ddof=1) * KM_PER_DEG

  pval_full_one = np.mean(boot_shifts_full <= 0.0)
  pval_full_two = 2.0 * min(pval_full_one, 1.0 - pval_full_one)

  line_mod_lower = np.percentile(boot_lines_mod, low_pct, axis=0)
  line_mod_upper = np.percentile(boot_lines_mod, high_pct, axis=0)


# ==============================================================================
# 6. ΣΧΕΔΙΑΣΗ ΓΡΑΦΗΜΑΤΟΣ
# ==============================================================================
if not df_final.empty:
  fig = plt.figure(figsize=(24, 14))
  plt.suptitle(
      "Storm Track Analysis (Clustered by SyCLoPS TID): "
      f" {REGION_TO_TEST.upper()} Crete (Threshold: {THRESHOLD_RAIN} mm/h |"
      f" Top-{TOP_N})",
      fontsize=22,
      fontweight="bold",
      y=0.98,
  )

  for i, (p, col) in enumerate(colors.items()):
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
          s=dot_size,
          alpha=dot_alpha,
          edgecolors="none",
          zorder=5,
      )
      n_tr = df_p["event"].nunique()
      ax.set_title(
          f"{p.replace('_', '–')}\nN={len(df_p)} timesteps ({n_tr} tracks)",
          fontsize=12,
          fontweight="bold",
      )

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
        color=colors[row["Period"]],
        markersize=16,
        markeredgecolor="k",
        zorder=10,
        label=row["Period"].replace("_", "–"),
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
        f"{row['Lat']:.2f}°N\n±{row['SE_km']:.1f}km",
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
          f"(Block bootstrap p {fmt_p(pval_full_two, 3)})"
      ),
  )

  ax_trend.plot(
      x[1:],
      np.poly1d(z_m)(x[1:]),
      color="red",
      linewidth=3,
      label=(
          f"Modern Trend (Post-1976): {total_shift_modern:+.3f}°\n"
          f"(Block bootstrap p {fmt_p(pval_mod_two, 3)})"
      ),
  )

  if len(df_final) == 4:
    ax_trend.fill_between(
        x[1:],
        line_mod_lower,
        line_mod_upper,
        color="red",
        alpha=0.15,
        label=f"Modern Trend 95% CI (Track Block Bootstrap Top-{TOP_N})",
    )

  for i, row in df_final.iterrows():
    ax_trend.errorbar(
        i,
        row["Lat"],
        yerr=row["SE_Lat"],
        fmt="s",
        color=colors[row["Period"]],
        markersize=12,
        markeredgecolor="k",
        capsize=8,
        capthick=1.5,
        elinewidth=1.5,
        zorder=10,
    )

  ax_trend.set_title(
      f"C. Latitudinal Trend Analysis (Top-{TOP_N} Clustered by TID)",
      fontsize=16,
      fontweight="bold",
  )
  ax_trend.set_xticks(x)
  ax_trend.set_xticklabels([s.replace("_", "–") for s in df_final["Period"]])
  ax_trend.set_ylabel("Mean Latitude (°N)", fontsize=12)
  ax_trend.grid(True, linestyle=":", alpha=0.6)
  ax_trend.legend(fontsize=10, loc="upper left")

  plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
  plt.show()

  # ==============================================================================
  # 7. ΕΚΤΥΠΩΣΗ ΤΕΛΙΚΩΝ ΣΤΑΤΙΣΤΙΚΩΝ
  # ==============================================================================
  print("\n" + "=" * 85)
  print(
      "ΣΤΑΤΙΣΤΙΚΑ CENTROIDS & SE [Clustered by SyCLoPS TID | Κατώφλι:"
      f" {THRESHOLD_RAIN} mm/h | Top-{TOP_N}]"
  )
  print("=" * 85)
  print(
      df_final[[
          "Period",
          "N_timesteps",
          "N_tracks",
          "Lat",
          "Lon",
          "Std_Lat",
          "SE_Lat",
          "SE_km",
      ]].to_string(index=False)
  )
  print("*(SE = Cluster-Robust Standard Error υπολογισμένο ανά SyCLoPS Track)*")

  if len(df_final) == 4:
    rate_mod_km_dec = (total_shift_modern * KM_PER_DEG) / DECADES_MODERN
    rate_full_km_dec = (total_shift_full * KM_PER_DEG) / DECADES_FULL

    print("\n" + "=" * 85)
    print(
        "ΣΤΑΤΙΣΤΙΚΗ ΕΓΚΥΡΟΤΗΤΑ ΤΑΣΕΩΝ (BOOTSTRAP 2000 RUNS ανά Track ID |"
        f" Top-{TOP_N})"
    )
    print("=" * 85)
    print("MODERN SHIFT (1976-2024 | Δt κέντρων περιόδων = 32.5 έτη):")
    print(
        "  • Μετατόπιση (Point Estimate) :"
        f" {total_shift_modern:+.4f}° ({total_shift_modern*KM_PER_DEG:+.2f} km)"
    )
    print(
        f"  • Ρυθμός Μετατόπισης          : {rate_mod_km_dec:+.2f} km / decade"
    )
    print(
        "  • 95% Confidence Interval     :"
        f" [{ci_mod_deg[0]:+.4f}°, {ci_mod_deg[1]:+.4f}°]  --> "
        f" [{ci_mod_km[0]:+.2f} km, {ci_mod_km[1]:+.2f} km]"
    )
    print(f"  • Τυπικό Σφάλμα Τάσης (SE)    : ±{se_mod_km:.2f} km")
    print(
        "  • Block Bootstrap p-value     : (H0: Shift = 0) "
        f"{fmt_p(pval_mod_two)}"
    )

    print("\nFULL SHIFT (1960-2024 | Δt κέντρων περιόδων = 48.5 έτη):")
    print(
        "  • Μετατόπιση (Point Estimate) :"
        f" {total_shift_full:+.4f}° ({total_shift_full*KM_PER_DEG:+.2f} km)"
    )
    print(
        f"  • Ρυθμός Μετατόπισης          : {rate_full_km_dec:+.2f} km / decade"
    )
    print(
        "  • 95% Confidence Interval     :"
        f" [{ci_full_deg[0]:+.4f}°, {ci_full_deg[1]:+.4f}°]  --> "
        f" [{ci_full_km[0]:+.2f} km, {ci_full_km[1]:+.2f} km]"
    )
    print(f"  • Τυπικό Σφάλμα Τάσης (SE)    : ±{se_full_km:.2f} km")
    print(
        "  • Block Bootstrap p-value     : (H0: Shift = 0) "
        f"{fmt_p(pval_full_two)}"
    )
    print("=" * 85)

  print(
      "Fallback σε απόσταση (χωρίς MFD):"
      f" {total_mfd_fallbacks} υποψήφια κέντρα σε"
      f" {total_mfd_fallback_events} χρονικά βήματα"
  )
  print("=" * 85)
