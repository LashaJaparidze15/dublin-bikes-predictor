"""Streamlit dashboard: live station map, predictions for one station, and live accuracy.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from bikes import store
from bikes.live import fetch_stations, fetch_status
from bikes.models import load_bundle
from bikes.predict import predict_now
from bikes.weather import fetch_forecast

st.set_page_config(page_title="Dublin Bikes predictor", page_icon="🚲", layout="wide")


@st.cache_data(ttl=300)
def live_data():
    status = fetch_status()
    stations = fetch_stations()
    # use the stored last 80 minutes for lag features when there is a database, else just now
    recent = store.recent_snapshots(minutes=80)
    now = status.rename(columns={"bikes": "num_bikes_available", "docks": "num_docks_available"})
    recent = pd.concat([recent, now]) if not recent.empty else now
    preds = predict_now(recent, fetch_forecast(), load_bundle())
    return status.merge(stations[["station_id", "name", "lat", "lon"]], on="station_id"), preds


status, preds = live_data()
st.title("🚲 Dublin Bikes availability predictor")
st.caption(f"Live data from Dublin City Council (CC BY 4.0), weather from Met Éireann. "
           f"Last reading {status['last_reported'].max().tz_convert('Europe/Dublin'):%H:%M}.")

if preds.empty:
    st.info("No stations are renting right now. Dublin Bikes closes overnight (about 00:30 to 05:00), "
            "so there is nothing to predict until it reopens.")
    st.stop()

tab_map, tab_station, tab_accuracy = st.tabs(["Map", "Station", "How accurate is it?"])

p30 = preds[preds["horizon_min"] == 30].set_index("station_id")
status = status.join(p30[["predicted_bikes", "p_empty", "p_full"]], on="station_id")

with tab_map:
    st.write("Colour shows the risk in **30 minutes**: red = likely empty, orange = likely full, "
             "grey = closed or no prediction.")
    fmap = folium.Map(location=[53.347, -6.26], zoom_start=14)
    for _, r in status.iterrows():
        if pd.isna(r["predicted_bikes"]):  # station closed / not renting, so the model skips it
            colour, tip = "grey", f"{r['name']}: not renting right now, no prediction"
        else:
            colour = "red" if r["p_empty"] > 0.5 else "orange" if r["p_full"] > 0.5 else "green"
            tip = (f"{r['name']}: {r['bikes']} bikes now, about {r['predicted_bikes']:.0f} in 30 min "
                   f"(empty {r['p_empty']:.0%}, full {r['p_full']:.0%})")
        folium.CircleMarker([r["lat"], r["lon"]], radius=7, color=colour, fill=True,
                            fill_opacity=0.8, tooltip=tip).add_to(fmap)
    st_folium(fmap, height=550, use_container_width=True)

with tab_station:
    names = status.sort_values("name").set_index("station_id")["name"]
    sid = st.selectbox("Station", names.index, format_func=lambda i: names[i].title())
    row = status.set_index("station_id").loc[sid]
    cols = st.columns(5)
    cols[0].metric("Bikes now", int(row["bikes"]))
    cols[1].metric("Free docks now", int(row["docks"]))
    mine = preds[preds["station_id"] == sid].set_index("horizon_min")
    for col, h in zip(cols[2:], [10, 30, 60]):
        if h in mine.index:
            col.metric(f"In {h} min", f"{mine.loc[h, 'predicted_bikes']:.0f}",
                       f"{mine.loc[h, 'predicted_bikes'] - row['bikes']:+.1f}")
    if pd.isna(row["p_empty"]):
        st.write("This station is not renting right now, so there is no prediction for it.")
    else:
        st.write(f"Chance it is **empty** in 30 minutes: **{row['p_empty']:.0%}**, "
                 f"**full**: **{row['p_full']:.0%}**")

with tab_accuracy:
    acc = store.prediction_accuracy(days=14)
    if acc.empty:
        st.info("Live accuracy tracking needs the Supabase database (set DATABASE_URL). "
                "Every collector run logs its predictions there and the prediction_accuracy view "
                "compares them with what actually happened.")
    else:
        acc["day"] = pd.to_datetime(acc["made_at"]).dt.tz_convert("Europe/Dublin").dt.date
        daily = (acc.groupby(["day", "horizon_min"])[["abs_error", "persistence_abs_error"]].mean()
                    .reset_index().melt(["day", "horizon_min"], var_name="model", value_name="MAE"))
        daily["model"] = daily["model"].map({"abs_error": "LightGBM", "persistence_abs_error": "Persistence"})
        st.plotly_chart(px.line(daily, x="day", y="MAE", color="model", facet_col="horizon_min",
                                title="Daily mean absolute error of live predictions (bikes)"),
                        use_container_width=True)
        st.caption(f"{len(acc):,} predictions checked against reality in the last 14 days.")
