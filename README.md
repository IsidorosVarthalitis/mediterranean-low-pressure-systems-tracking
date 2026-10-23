# mediterranean-low-pressure-systems-tracking
An automated, objective computational framework developed in Python examining whether a northward shift of Mediterranean low–pressure systems causing precipitation events over the island of Crete is already evident. Hourly ERA5 meteorological reanalysis fields (0.25°×0.25°) were used along with 3–hourly synoptic track records from the SyCLoPS database across a 65–year climatological baseline (1960–2024). Candidate low–pressure centers within the Eastern Mediterranean (28°–42°N, 18°–32°E) were synchronized through nearest–neighbour temporal rounding to the 3–hourly track steps (maximum offset 1 h) and ranked using a hybrid score combining latitude–corrected (cos φ) distance to Crete with the vertically integrated moisture flux divergence (VIMDF) averaged over a 5°×5° window around each center, moisture convergence (MFD < 0) reduces the score, favouring moisture–converging systems, while divergence leaves it unchanged.
---

## Key Features & Statistical Pipeline
- **Spatiotemporal Coupling:** Automates matching between ERA5 high-resolution precipitation data and synoptic system tracks points.
- **Hybrid Score:** Balances spatial distance to Crete (cos(lat)-corrected) with moisture convergence. Cases where MFD is unavailable fall back to distance only and are counted and reported.
- **Track-Aware Uncertainty (SyCLoPS Track ID, `TID`):** Consecutive 3-hourly positions of the same cyclone are not independent, so all uncertainty estimates treat each SyCLoPS track as one cluster:
  - Cluster-robust (CR1) standard error of each period's mean latitude.
  - Block bootstrap resampling whole tracks (2,000 iterations) for the full (1960–2024) and modern (1976–2024) trends, with 95% Confidence Intervals and two-sided p-values (H0: shift = 0).
  - Shift rates reported in km/decade.

---

## Environment & Dependencies
Developed and tested in **Python 3.13** (using the **Spyder IDE**).

The following libraries are required:
- `numpy`
- `pandas`
- `xarray`
- `cartopy`
- `matplotlib`
- `pyarrow` (required engine for reading `.parquet` files with column filtering)
- `netCDF4`

```bash
##To install the dependencies:

pip install -r requirements.txt

##Expected Directory & Data Setup
The script expects datasets to be located inside a local data/ directory relative to the repository root:

Plaintext
.
├── data/
│   ├── SyCLoPS_classified_ERA5_1940_2024.parquet
│   ├── era5_rain_<year>.nc
│   └── era5_mfd_<year>.nc
├── mediterranean-low-pressure-systems-tracking.py
├── requirements.txt
└── README.md

##Configuration & Usage
Key parameters can be adjusted directly in mediterranean-low-pressure-systems-tracking.py:

THRESHOLD_RAIN: Hourly precipitation threshold in mm/h (default: 5).

TOP_N: Number of candidate low-pressure centers retained per event (default: 1 for Top-1, 3 for Top-3).

REGION_TO_TEST: Target spatial domain ("all", "west", or "east" Crete).

##Execution
Run the script directly from the terminal or your preferred Python IDE (e.g., Spyder, VS Code):
python mediterranean-low-pressure-systems-tracking.py

##Author & Institutional Affiliation
Author: Isidoros Varthalitis

Academic Context: Diploma Thesis Research (Integrated Master)

Institution: Technical University of Crete (TUC)

Department: School of Environmental Engineering
