# mediterranean-low-pressure-systems-tracking
An automated, objective computational framework developed in Python examining whether a northward shift of Mediterranean low–pressure systems causing precipitation events over the island of Crete is already evident. Hourly ERA5 meteorological reanalysis fields (0.25°×0.25°) were used along with 3–hourly synoptic track records from the SyCLoPS database across a 65–year climatological baseline (1960–2024). Candidate low–pressure centers within the Eastern Mediterranean (28°–42°N, 18°–32°E) were synchronized through nearest–neighbour temporal rounding to the 3–hourly track steps (maximum offset 1 h) and ranked using a hybrid score combining latitude–corrected (cos φ) distance to Crete with the vertically integrated moisture flux divergence (VIMDF) averaged over a 5°×5° window around each center, moisture convergence (MFD < 0) reduces the score, favouring moisture–converging systems, while divergence leaves it unchanged.

---

## Key Features & Statistical Pipeline
- **Spatiotemporal Coupling:** Automates matching between ERA5 high-resolution precipitation data and synoptic system track points.
- **Hybrid Score:** Balances spatial distance to Crete (cos(lat)-corrected) with moisture convergence:
  `score = dist / (1 + max(0, −MFD × 1000))`, lower is better. The relative weight of the convergence term depends on the units of the MFD field (the script reads the variable `vimdf`, or `mfd` if `vimdf` is absent).
  Cases where MFD is unavailable fall back to distance only; they are counted and reported both as candidate centres and as 3-hourly time steps.
- **Track-Aware Uncertainty (SyCLoPS Track ID, `TID`):** Consecutive 3-hourly positions of the same cyclone are not independent, so all uncertainty estimates treat each SyCLoPS track as one cluster:
  - Cluster-robust (CR1) standard error of each period's mean latitude.
  - Block bootstrap resampling whole tracks (2,000 iterations) for the full (1960–2024) and modern (1976–2024) trends, with 95% Confidence Intervals and two-sided p-values (H0: shift = 0). A p-value of zero in the bootstrap is reported as `p < 1/2000`.
  - Shift rates reported in km/decade.

---

## Environment & Dependencies
Developed and tested in **Python 3.13** (using the **Spyder IDE**).

Required libraries:
- `numpy`
- `pandas`
- `xarray`
- `netCDF4`
- `pyarrow` (required engine for reading `.parquet` files with column filtering)
- `matplotlib`
- `cartopy`

To install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Expected Directory & Data Setup
The script expects the datasets inside a local `data/` directory relative to the repository root:

```text
.
├── data/
│   ├── SyCLoPS_classified_ERA5_1940_2024.parquet
│   ├── era5_rain_<year>.nc
│   └── era5_mfd_<year>.nc
├── mediterranean-low-pressure-systems-tracking.py
├── requirements.txt
└── README.md
```

- `era5_rain_<year>.nc`: hourly ERA5 total precipitation `tp` (m), one file per year.
- `era5_mfd_<year>.nc`: ERA5 moisture flux divergence (`vimdf` or `mfd`), one file per year, available at the 3-hourly track times.
- The SyCLoPS parquet file must contain the columns `LAT`, `LON`, `ISOTIME` (timestamp) and `TID`.

---

## Configuration & Usage
Key parameters can be adjusted directly in `mediterranean-low-pressure-systems-tracking.py`:

| Parameter | Description | Default |
|---|---|---|
| `THRESHOLD_RAIN` | Hourly precipitation threshold in mm/h | `5` |
| `TOP_N` | Number of candidate low-pressure centers retained per event (`1` for Top-1, `3` for Top-3) | `1` |
| `REGION_TO_TEST` | Target spatial domain: `"all"`, `"west"` or `"east"` Crete | `"all"` |

Precipitation events are detected when any ERA5 grid cell in the box 34.5°–36.0°N, 23.5°–26.5°E exceeds `THRESHOLD_RAIN` (no land mask is applied, so nearby sea cells are included). The `west` and `east` sub-regions are 23.5°–25.0°E and 25.0°–26.5°E respectively.

---

## Execution
Run the script directly from the terminal or your preferred Python IDE (e.g., Spyder, VS Code):

```bash
python mediterranean-low-pressure-systems-tracking.py
```

---

## Notes & Limitations
- With `TOP_N = 3`, the three centres selected at the same time step belong to different tracks but to the same precipitation event; possible correlation between them is not modelled by the track-level bootstrap, so Top-3 confidence intervals may be slightly narrow.
- The 1960–1975 period lies entirely before the satellite era (1979), when ERA5 assimilates fewer observations; the modern trend (1976–2024) is less affected by this.
- Because centres are selected by proximity to Crete, the estimated shift describes the systems that produce rainfall over Crete and may be smaller than a shift of the full cyclone population.

---

## Author & Institutional Affiliation
- **Author:** Isidoros Varthalitis
- **Academic Context:** Diploma Thesis Research (Integrated Master)
- **Institution:** Technical University of Crete (TUC)
- **Department:** School of Environmental Engineering
