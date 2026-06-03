#from quartz_solar_forecast import run_forecast
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

selected_gsps = [152, 199, 202, 324]

arima_predictions = pd.read_feather("/home/andreacatucci/all_predictions.feather")
arima_predictions.columns = [col.lower() for col in arima_predictions.columns]

truth = (
    arima_predictions[
        (arima_predictions['gsp_id'].isin(selected_gsps)) &
        (pd.to_datetime(arima_predictions['datetime_hour']).dt.year == 2022)
    ][["gsp_id", "datetime_hour", "generation_mw"]]
)

capacity_data = pd.read_feather('/home/andreacatucci/capacity_by_month.feather')

capacity_data = capacity_data[capacity_data['gsp_id'].isin(selected_gsps)]

if 'year' not in capacity_data.columns:
    capacity_data['year'] = pd.to_datetime(capacity_data['install_month']).dt.year
if 'month' not in capacity_data.columns:
    capacity_data['month'] = pd.to_datetime(capacity_data['install_month']).dt.month

gsp_capacity_dicts = (
    capacity_data
    .groupby('gsp_id')
    .apply(lambda df: {(row['year'], row['month']): row['total_capacity_mwp'] *1000
                       for _, row in df.iterrows()})
    .to_dict()
)

gsp_capacity_dicts = (
    capacity_data
    .groupby('gsp_id')
    .apply(lambda df: {
        (row['year'], row['month']): row['total_capacity_mwp'] * 1000
        for _, row in df.iterrows()
    }, include_groups=False, )
    .to_dict()
)

# Prep Quartz-compatible meta
gsp_meta = pd.DataFrame({
    'gsp_id': selected_gsps,
    'latitude': [53.7, 55.9, 56, 52.6],
    'longitude': [-1.27, -4.35, -2.57, 1.27],
})

# Init list for results
results = []

for _, row in tqdm(gsp_meta.iterrows(), total=gsp_meta.shape[0], desc="GSP Loop"):
    gsp_id = row.gsp_id
    lat = row.latitude
    lon = row.longitude
    cap_dict = gsp_capacity_dicts[gsp_id]
    print("Started loop for gsp: ", gsp_id)

    # Get all test timestamps for this GSP and ensure they are sorted
    timestamps = truth[truth['gsp_id'] == gsp_id]['datetime_hour'].unique()
    timestamps = pd.to_datetime(sorted(timestamps))

    for ts in tqdm(timestamps, desc=f"GSP {gsp_id}", leave=False):
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)

        year_month = (ts.year, ts.month)
        capacity_kwp = cap_dict.get(year_month)
        if capacity_kwp is None:
            continue

        site = PVSite(latitude=lat, longitude=lon, capacity_kwp=capacity_kwp)
        try:
            df_q = run_forecast(site=site, ts=ts, nwp_source="icon", model="gb")
        except Exception as e:
            print(f"Error on {ts} for GSP {gsp_id}: {e}")
            continue

        print("columns: ", df_q.columns)  

        # Index is already datetime — select the next 4 time steps (ts + 15m to ts + 1h)
        next_4_steps = df_q.loc[ts + pd.Timedelta(minutes=15): ts + pd.Timedelta(minutes=60)]

        if next_4_steps.empty or next_4_steps.shape[0] < 4:
            print(f"[SKIPPED] Not enough forecast points after {ts} for GSP {gsp_id}")
            continue

        forecast_sum = next_4_steps['power_kw'].sum()

        forecast_row = pd.DataFrame({
            "forecast_time": [ts + pd.Timedelta(hours=1)],
            "power_kw": [forecast_sum],
            "gsp_id": [gsp_id],
            "ts": [ts]
        })

        results.append(forecast_row)
        print(f"[SUCCESS] Forecast saved for GSP {gsp_id} at {ts + pd.Timedelta(hours=1)}. Forecasted value: {forecast_sum} kW")

# Safety check before concatenation
if len(results) == 0:
    raise RuntimeError("No forecasts generated. Please check your data and code.")

df_forecasts = pd.concat(results, ignore_index=True)
df_forecasts = df_forecasts.rename(columns={'power_kw': 'quartz_forecast'})

df_forecasts['forecast_time'] = pd.to_datetime(df_forecasts['forecast_time']).dt.tz_localize(None)
truth['datetime_hour'] = pd.to_datetime(truth['datetime_hour']).dt.tz_localize(None)

df_merge = df_forecasts.merge(
    truth[['datetime_hour', 'gsp_id', 'generation_mw']],
    left_on=['forecast_time', 'gsp_id'],
    right_on=['datetime_hour', 'gsp_id'],
    how='inner'
)

df_merge['error'] = df_merge['generation_mw'] - df_merge['quartz_forecast']

mae = mean_absolute_error(df_merge['generation_mw'], df_merge['quartz_forecast'])
rmse = mean_squared_error(df_merge['generation_mw'], df_merge['quartz_forecast'])

print(f"\nQuartz Forecast Performance:")
print(f"MAE: {mae:.3f} KW")
print(f"RMSE: {rmse:.3f} KW")

df_merge.to_feather("/home/andreacatucci/quartz_vs_truth.feather")