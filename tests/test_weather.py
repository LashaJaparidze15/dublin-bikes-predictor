from bikes.weather import parse_forecast

XML = """<weatherdata><product>
<time from="2026-11-03T01:00:00Z" to="2026-11-03T01:00:00Z"><location>
  <temperature value="10.6"/><windSpeed mps="1.9"/><humidity value="72.1"/></location></time>
<time from="2026-11-03T00:00:00Z" to="2026-11-03T01:00:00Z"><location>
  <precipitation value="0.4"/></location></time>
<time from="2026-11-03T00:00:00Z" to="2026-11-03T06:00:00Z"><location>
  <precipitation value="3.0"/></location></time>
</product></weatherdata>"""


def test_parse_forecast_pairs_instant_and_hourly_rain():
    df = parse_forecast(XML)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["temp_c"] == 10.6 and row["wind_ms"] == 1.9
    assert row["rain_mm"] == 0.4  # the 6-hour total is ignored
