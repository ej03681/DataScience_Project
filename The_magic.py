import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import mplcursors
import warnings

# Ask user for a target city
city_name = input("Enter a city name (e.g., Austin, Paris, Tokyo): ")

# Query the Open-Meteo Geocoding API to get the city's coordinates
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
geo_params = {"name": city_name, "count": 1}

print(f"Looking up coordinates for '{city_name}'...")
geo_response = requests.get(GEO_URL, params=geo_params)

if geo_response.status_code == 200 and geo_response.json().get("results"):
    # Extract location details from the first search result
    location = geo_response.json()["results"][0]
    lat = location["latitude"]
    lon = location["longitude"]
    resolved_name = f"{location['name']}, {location.get('country', '')}"

    print(f"Found: {resolved_name} (Lat: {lat}, Lon: {lon})")


    # Fetch weather data using the resolved coordinates
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "forecast_days": 1,
        "timezone": "auto"
    }

    weather_response = requests.get(WEATHER_URL, params=weather_params)

    if weather_response.status_code == 200:
        # 4. Process data into a Pandas DataFrame
        data = weather_response.json()["hourly"]
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.columns = ['Time', 'Temperature (°F)']



        df['Hour'] = df['Time'].dt.strftime('%I %p')
        df_12h = df.head(12).copy()

        forecast_date = df_12h['Time'].iloc[0].strftime('%B %d, %Y')

        # Build the Seaborn plot
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="darkgrid")




        line_plot = sns.lineplot(data=df_12h, x='Hour', y='Temperature (°F)', color='dodgerblue', linewidth=2, marker='o', markersize=8)

        cursor = mplcursors.cursor(line_plot, hover=True)
        warnings.filterwarnings("ignore", category=UserWarning, module="mplcursors")

        # Format the tooltip popup string dynamically to display the precise degrees
        @cursor.connect("add")
        def on_add(sel):
            # sel.target[1] extracts the exact Y value (temperature) from the point
            sel.annotation.set_text(f"{sel.target[1]:.1f}°F")
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, boxstyle="round,pad=0.5")

        plt.title(f"12 Hour Forecast {forecast_date} in {resolved_name}", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Hour of the Day", fontsize=12)
        plt.ylabel("Temperature (°F)", fontsize=12)
        plt.xticks(rotation=2)
        plt.gcf().canvas.manager.set_window_title('Live 12 Hour Forecast')
        plt.tight_layout()

        plt.show()
    else:
        print("Failed to pull weather data.")
else:
    print(f"Could not find any city data matching '{city_name}'. Check your spelling!")
