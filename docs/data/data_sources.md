# Data sources

Data are **not** stored in git. This file is the registry of everything the pipeline uses.
Check every licence before redistributing derived products.

| Dataset | Use | Resolution | Access | Licence |
|---------|-----|-----------|--------|---------|
| HLS v2 (HLSL30/HLSS30) | Optical input (Prithvi native) | 30 m, 2–3 days | NASA CMR-STAC `LPCLOUD` (Earthdata login) | Open (NASA) |
| Sentinel-2 L2A | Optical input, red-edge | 10–20 m | Planetary Computer `sentinel-2-l2a` | Copernicus open |
| Sentinel-1 RTC | SAR input | 10 m | Planetary Computer `sentinel-1-rtc` | Copernicus open |
| Copernicus DEM GLO-30 | Terrain | 30 m | Planetary Computer `cop-dem-glo-30` | Copernicus DEM licence |
| CHELSA v2.1 | Climate normals | ~1 km | chelsa-climate.org | CC BY 4.0 |
| TerraClimate | Monthly P, PET, water balance | ~4 km | GEE / THREDDS | CC0 |
| CHIRPS v2 | Precipitation | ~5 km | GEE / UCSB | Public domain |
| ERA5-Land | Meteorology / forcing | ~9 km | Copernicus CDS | Copernicus licence |
| ESA WorldCover v200 | Reference labels (coarse) | 10 m | Planetary Computer / Zenodo | CC BY 4.0 |
| MapBiomas (Amazonía, PE, BO, EC, CO) | Reference labels | 30 m | GEE / platform | CC BY-SA 4.0 |
| National ecosystem maps (EC, PE, BO) | Reference labels | varies | Ministries | **TODO** check per map |
| Glacier inventories (RGI 7, national) | Snow/ice labels | vector | GLIMS / NSIDC | **TODO** |
| Field plots (BRIDGE) | Labels + validation | points | Internal → to be published | **TODO** |

## Local layout
```
data/
├── raw/         # untouched downloads (per source/site/date)
├── interim/     # masked, composited, reprojected
├── processed/   # analysis-ready stacks and chips (data/processed/chips/*.npz)
├── labels/      # harmonised reference labels (BRIDGE legend ids)
└── external/    # third-party maps used for comparison
```
