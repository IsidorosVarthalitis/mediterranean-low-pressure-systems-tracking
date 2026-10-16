# mediterranean-low-pressure-systems-tracking
An automated, objective computational framework developed in Python examining whether a northward shift of Mediterranean low–pressure systems causing precipitation events over the island of Crete is already evident. Hourly ERA5 meteorological reanalysis fields (0.25°×0.25°) were used along with 3–hourly synoptic track records from the SyCLoPS database across a 65–year climatological baseline (1960–2024). Candidate low–pressure centers within the Eastern Mediterranean (28°–42°N, 18°–32°E) were synchronized through nearest–neighbour temporal rounding (Δt≤±1.5h) and ranked using a hybrid score combining Euclidean proximity to Crete with vertically integrated moisture flux divergence (under convergence condition, MFD<0).
---

## Key Features & Statistical Pipeline
- **Spatiotemporal Coupling:** Automates matching between ERA5 high-resolution precipitation extremes and synoptic tracking points.
- **Hybrid Optimization Score:** Balances spatial distance to Crete with moisture convergence (VIMDF).
- **Statistical Significance Testing:**
  - Non-parametric bootstrapping (2,000 iterations) for trend estimation and 95% Confidence Intervals (CI).
  - Permutation testing (1,000 resamples) for rigorous null-hypothesis validation.

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

##To install the dependencies:
```bash
pip install -r requirements.txt

##Expected Directory & Data Setup
The script expects datasets to be located inside a local data/ directory relative to the repository root:

Plaintext
.
├── data/
│   ├── SyCLoPS_classified_ERA5_1940_2024.parquet
│   ├── era5_rain_<year>.nc
│   └── era5_mfd_<year>.nc
├── track_analysis.py
├── requirements.txt
└── README.md

##Configuration & Usage
Key parameters can be adjusted directly in track_analysis.py:

THRESHOLD_RAIN: Hourly precipitation threshold in mm/h (default: 5).

TOP_N: Number of candidate low-pressure centers retained per event (default: 1 for Top-1, 3 for Top-3).

REGION_TO_TEST: Target spatial domain ("all", "west", or "east" Crete).

##Execution
Run the script directly from the terminal or your preferred Python IDE (e.g., Spyder, VS Code):

Bash
python track_analysis.py

##Author & Institutional Affiliation
Author: Isidoros Varthalitis

Academic Context: Diploma Thesis Research (Integrated Master)

Institution: Technical University of Crete (TUC)

Department: School of Environmental Engineering
