# Austin Year-over-Year Weather Analysis & Machine Learning Forecast

An offline-first data science project that builds a localized weather database for Austin, TX using the Open-Meteo API, traces year-over-year climate anomalies, and applies a scikit-learn predictive model to forecast rolling temperature curves.

## 📁 Project Directory Breakdown

The project repository is partitioned into modular data engineering, analytics, and modeling components to guarantee computational stability and eliminate live API server rate limits:

*   **`The_magic.py`**:  The original project. User selects city and application connects to Geocoding API in Open-Meteo to collect coordinates. Then displays 12-hour forecast chart.
*   **`download_data.py`**:  The data acquisition core. Connects to the Open-Meteo API engine to query a multi-year chronological weather profile and outputs a consolidated, localized cache file named `austin_weather_history.csv`.
*   **`Compare_Four_Years.py`**: Reads the offline local dataset into a Pandas DataFrame and deploys Seaborn to map overlapping 12-hour temperature trajectories across custom-built historical baselines with interactive `mplcursors` hover badges.
*   **`Predict_Weather.py`**: Transmutes time-series rows into a supervised learning matrix via lagged window features (`shift(1)` and `shift(2)`). Trains a scikit-learn `LinearRegression` model on historical seasons to forecast upcoming local conditions.
*   **`austin_weather_history.csv`**: The local dataset cache holding clean structured columns for `Hour`, `Temperature (°F)`, and `Year`.

## ⚙️ Data Engineering Pipeline

To avoid script crashes from server-side rate limits or data stream drops, the analytics engine handles dates natively:
1.  **Chronological Data Stacking**: Multi-year tables are sorted sequentially to push data offsets safely into past baselines, keeping target prediction records completely intact.
2.  **Feature Engineering Lag Matrix**: Shifting time intervals creates explicit tracking metrics (`Temp_Lag_1` and `Temp_Lag_2`) so scikit-learn can extract directional heating and cooling coefficients mathematically.

## 🛠️ Installation & Execution

1. Clone this repository to your local workstation.
2. Install the production data science package stack:
   ```bash
   pip install pandas requests seaborn scikit-learn mplcursors
   ```
3. Run (`The_magic.py`) once prompted to enter city, enter city you would like to look at. 12 hour forecast chart is executed.
4. Run your downloader script (`download_data.py`) once to initialize the local CSV data file cache.
5. Run `Compare_Four_Years.py` to launch the historical chart interface or execute `Predict_Weather.py` to evaluate the machine learning predictions.