import pandas as pd
import requests
import time

# Hardcoded project configuration settings for Austin, TX
lat = 30.2672
lon = -97.7431
location_name = "Austin, United States"

# FIX: Force a completed past calendar date window to guarantee data exists
target_date_part = "09-18"
display_date = "September 18"

# The target year sequences we want to capture
target_years = [2010, 2014, 2018, 2022, 2026]
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

print(f"🚀 Initializing API Downloader for {location_name}...")
all_years_data = []

for year in target_years:
    # Construct an error-free, fully historical YYYY-MM-DD compliant tracking string
    api_date = f"{year}-{target_date_part}"
    print(f" -> Connecting to archive API for: {api_date}...")

    hist_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": api_date,
        "end_date": api_date,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "America/Chicago"
    }

    # Safe 1-second delay to avoid hitting server automated request filters
    time.sleep(1)
    h_res = requests.get(HISTORICAL_URL, params=hist_params)

    if h_res.status_code == 200:
        try:
            hist_json = h_res.json()
            if "hourly" in hist_json:
                h_data = hist_json["hourly"]
                df_h = pd.DataFrame(h_data).head(12).copy()
                df_h['time'] = pd.to_datetime(df_h['time'])

                # Standardize tracking row shapes
                df_h['Hour'] = df_h['time'].dt.strftime('%I %p')
                df_h['Temperature (°F)'] = df_h['temperature_2m']
                df_h['Year'] = str(year)

                all_years_data.append(df_h[['Hour', 'Temperature (°F)', 'Year']])
        except ValueError:
            print(f" ❌ Error processing returned JSON data for year {year}.")
    else:
        print(f" ❌ Error: API server rejected query for year {year}. HTTP {h_res.status_code}")

# Consolidate our matrices and output to a local CSV file
if all_years_data:
    master_df = pd.concat(all_years_data, ignore_index=True)
    master_df.to_csv('austin_weather_history.csv', index=False)
    print(f"\n✅ Success! All records downloaded and saved to 'austin_weather_history.csv'.")
else:
    print("\n❌ Failed: No weather data could be collected from the server.")

