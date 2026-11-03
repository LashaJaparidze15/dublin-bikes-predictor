# Dublin Bikes availability predictor

Data Science & Machine Learning portfolio, Lasha Japaridze (C00303432).

A scheduled job collects live Dublin Bikes station data every 5 minutes into Postgres. Models
trained on 18 months of history (January 2025 to June 2026) joined with Met Éireann weather
answer *"will station X have a bike in 30 minutes?"*. Predictions are served through a small
API and a Streamlit dashboard, and every live prediction is logged so its accuracy can be
checked against what really happened.

**Live dashboard:** https://dublin-bikes-predictor-xemaugbi5xtyxcss4hkhzs.streamlit.app/
(it sleeps when unused and takes about 30 seconds to wake; the scheme itself closes roughly 00:30 to 05:00)

## Questions

1. How many bikes will be at station X in Y minutes? (regression, Y = 10, 30, 60)
2. Will station X be empty or full in 30 minutes? (classification)
3. How much does rain reduce usage?
4. Is there a commuter pattern, with outer stations emptying in the morning and the city centre filling up?

## Results so far

Trained on January 2025 to March 2026, tested on April to June 2026 (never used for any choice).

**Q1. Bikes in Y minutes** (mean absolute error, bikes)

| Model | 10 min | 30 min | 60 min |
|---|---|---|---|
| Persistence ("nothing changes") | **0.47** | 1.06 | 1.70 |
| Historical average | 5.42 | 5.42 | 5.41 |
| Ridge | 0.53 | 1.17 | 1.87 |
| LightGBM (predicts the change, Huber loss) | 0.50 | **0.99** | **1.48** |

Persistence is unbeaten at 10 minutes. LightGBM is 6% better at 30 minutes and 13% better at 60.

**Q2. Empty / full in 30 minutes.** LightGBM PR-AUC 0.88 for empty and 0.80 for full, against
0.63 and 0.51 for "same as now". Catching stations that are about to run out is the hard part
and is the focus for the final version.

**Q3. Rain.** Compared with a dry hour at the same time of day, day type and month, network
activity drops about 12% in light rain, 28% in moderate rain and 36% in heavy rain. But weather
adds nothing to the 30-minute prediction, because the current count already reflects it.

**Q4. Commuter pattern.** Yes. K-means on weekday profiles finds a group of stations that fills
up in the morning (the south-east office district: Merrion Square, St Stephen's Green East) and
one that empties (mainly the rail hubs: Heuston and the area around Connolly). People arrive by
train and ride the last stretch to work. Map: `notebooks/station_clusters_map.html`.

## Repository layout

```
bikes/                 shared, tested code (used by notebooks, collector, dashboard and API)
  config.py            URLs, paths, constants
  history.py           download monthly history CSVs -> parquet
  weather.py           Met Éireann hourly history + forecast parser
  live.py              live GBFS feed
  features.py          10-minute grid, lags, time and weather features, targets
  models.py            baselines, Ridge, LightGBM regressor and classifier, metrics
  train.py             builds datasets, trains and saves models/bundle.joblib
  predict.py           live predictions from the saved bundle
  store.py             Supabase Postgres, or local parquet when DATABASE_URL is not set
  collector.py         one collection run (GitHub Actions calls this every 5 minutes)
notebooks/
  01_data_collection_and_cleaning.ipynb
  02_exploratory_analysis.ipynb      commuter pattern (k-means), rain effect
  03_modelling.ipynb                 baselines vs models, 10/30/60 min, empty/full
app/streamlit_app.py   dashboard
api/main.py            FastAPI JSON API
sql/schema.sql         Supabase tables and the prediction_accuracy view
tests/                 pytest
.github/workflows/collect.yml
LOG.md                 weekly log
```

## Running it

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt   # or python -m venv + pip

python -m bikes.history 2025-01 2026-06     # ~18 months of history (about 3 GB download, 30 MB stored)
python -c "from bikes.weather import download_hourly; download_hourly()"
python -m bikes.train                       # evaluates, then saves models/bundle.joblib
pytest -q

python -m bikes.collector                   # one live collection + prediction run
streamlit run app/streamlit_app.py
uvicorn api.main:app                        # GET /predictions, /predictions/{station_id}
```

Without `DATABASE_URL` everything runs locally and live data goes to `data/live/*.parquet`.

### Live collection on Supabase + GitHub Actions

1. Create a Supabase project and run `sql/schema.sql` in its SQL editor.
2. Add the connection string (Session pooler URI) as a repository secret called `DATABASE_URL`.
3. The `collect` workflow then runs every 5 minutes. GitHub runs scheduled jobs late or skips
   them at busy times, so the real interval is irregular. The 10-minute grid handles that.
4. On Streamlit Community Cloud, point the app at `app/streamlit_app.py` and add the same
   `DATABASE_URL` under the app's secrets.

## Data and licences

* Dublin City Council, *Dublinbikes API* and monthly historical station data, data.smartdublin.ie / data.gov.ie, CC BY 4.0.
* Met Éireann, *Dublin Airport Hourly Data*, data.gov.ie, CC BY 4.0.
* Met Éireann point forecast API (openaccess.pf.api.met.ie), Met Éireann open data licence.
