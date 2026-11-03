# Weekly log

## Week 1: Wed 14 to Tue 20 Oct 2026
- Spec was in (9 Oct), so started on the data sources.
- Tested the live GBFS feed by hand. 115 stations, a reading every 5 to 10 minutes. Station IDs
  in the live feed match the ones in the history files, so live and historical data can be joined.
- Looked through the monthly history files on data.smartdublin.ie. Files before May 2024 use a
  different layout, so I decided to use January 2025 onwards only (18 months, all one format).
- Each monthly CSV is over 100 MB because station name, address and coordinates repeat on every
  row. Tried keeping only the columns that change and saving as parquet, which makes it tiny.
- Downloaded the Met Éireann hourly file for Dublin Airport. It starts with a text key before the
  real header, so it needs a custom loader. Wind is in knots there but m/s in the forecast, so I
  convert it.
- Problem: Met Éireann's forecast API kept timing out. Parked it for now.

## Week 2: Wed 21 to Tue 27 Oct 2026
- Spent most of the week on data quality in notebooks.
- Readings are irregular, so I tried resampling onto a fixed grid. 10 minutes worked best:
  5 minutes left too many empty slots.
- Found that closed / not-renting stations report 0 bikes. Left in, they would teach a model
  that "closed" means "empty", so they get dropped.
- Spotted big jumps of 8+ bikes in one slot. These are the rebalancing vans. Decided to keep
  them because they are real, but they will be hard for any model.
- Worked out the time zone. The timestamps have no zone, but the morning rush sits at 08:00 in
  winter and 07:00 in summer, so they must be UTC. Hour features now use Irish local time.
- First rough feature ideas: current bikes, lags, hour of day, weekday, weather.
- Quick first test on two months: the "nothing changes" baseline is surprisingly strong at
  30 minutes. Something to deal with properly next week.

## Week 3: Wed 28 Oct to Tue 3 Nov 2026 (draft portfolio due 3 Nov)
- Moved the notebook code into a proper `bikes/` package so the collector, notebooks and
  dashboard share one tested version. Added pytest tests for the grid, lags, targets, time zone
  and forecast parser.
- Got the forecast API working: HTTPS hangs but plain HTTP is fine.
- Built the full pipeline over 18 months: 10-minute grid, lag / time / weather features and
  targets for 10, 30 and 60 minutes ahead.
- Models: persistence, historical average, Ridge, LightGBM, plus a LightGBM empty/OK/full
  classifier. Plain LightGBM did not beat persistence on MAE, so it now predicts the *change* in
  bikes with Huber loss. That fixed it at 30 and 60 minutes. At 10 minutes persistence still
  wins on MAE, which is a result worth reporting rather than hiding.
- Validation is time-based only: April to June 2026 held out, TimeSeriesSplit inside training.
- Wrote the live collector (GBFS to Postgres, with a local parquet fallback), the GitHub Actions
  schedule, the Supabase schema with a `prediction_accuracy` view, a Streamlit dashboard and a
  small FastAPI endpoint.
- Next: set up Supabase and the repo secret so live collection starts, then a notebook on how
  accurate the live predictions are over time.
