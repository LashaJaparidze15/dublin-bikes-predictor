"""Paths and constants shared across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"      # one parquet file per month of station snapshots
WEATHER_DIR = DATA_DIR / "weather"
MODELS_DIR = ROOT / "models"

# Live feed (GBFS v2.3, run by JCDecaux for Dublin City Council)
GBFS_STATUS_URL = "https://api.cyclocity.fr/contracts/dublin/gbfs/v2/station_status.json"
GBFS_INFO_URL = "https://api.cyclocity.fr/contracts/dublin/gbfs/v2/station_information.json"

# Historical monthly CSVs are listed on this CKAN dataset
CKAN_PACKAGE_URL = "https://data.smartdublin.ie/api/3/action/package_show?id=dublinbikes-api"

# Met Éireann hourly observations for Dublin Airport (station 532)
WEATHER_HOURLY_URL = "https://clidata.met.ie/cli/climate_data/webdata/hly532.csv"
# Met Éireann point forecast (XML), city centre coordinates
# (the HTTPS endpoint hangs, plain HTTP works)
FORECAST_URL = "http://openaccess.pf.api.met.ie/metno-wdb2ts/locationforecast?lat=53.3498;long=-6.2603"

# Snapshots are resampled onto this grid so every station has one row per slot
SLOT_MINUTES = 10
TIMEZONE = "Europe/Dublin"
