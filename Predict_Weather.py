import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings

# Mute standard background warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("🤖 Initializing scikit-learn Weather Predictor...")

# 1. Load your saved local weather dataset
try:
    master_df = pd.read_csv('austin_weather_history.csv')

    # FIX: Force the Year column to integers to prevent data filtering dropouts
    master_df['Year'] = master_df['Year'].astype(str).str.extract(r'(\d+)').astype(int)

except FileNotFoundError:
    print("❌ Error: 'austin_weather_history.csv' not found! Run your downloader script first.")
    exit()

# 2. FEATURE ENGINEERING: Transform time sequences into a ML-ready matrix
df_ml = master_df.copy()
df_ml['Temp_Lag_1'] = df_ml['Temperature (°F)'].shift(1)
df_ml['Temp_Lag_2'] = df_ml['Temperature (°F)'].shift(2)

print(df_ml[['Year', 'Hour', 'Temperature (°F)', 'Temp_Lag_1', 'Temp_Lag_2']])
print("--------------------------------------------\n")
# Drop missing rows created by shifting data windows
df_ml = df_ml.dropna()

# 3. SPLIT DATA INTO TRAINING AND TESTING SETS (Using pure integers)
train_data = df_ml[df_ml['Year'] != 2026]
test_data = df_ml[df_ml['Year'] == 2026]

# Double check that we actually have samples to train on
if train_data.empty or test_data.empty:
    print(f"❌ Error: Dataset filtering issue. Train rows: {len(train_data)}, Test rows: {len(test_data)}")
    print("Available years in CSV:", master_df['Year'].unique())
    exit()

# Define features (X) and target value to predict (y)
features = ['Temp_Lag_1', 'Temp_Lag_2']
X_train = train_data[features]
y_train = train_data['Temperature (°F)']

X_test = test_data[features]
y_test = test_data['Temperature (°F)']

# 4. INITIALIZE AND TRAIN THE LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Machine Learning Model training complete.")

# 5. PREDICT THE NEXT 12 HOURS
test_data = test_data.copy()
test_data['Predicted Temperature'] = model.predict(X_test)

# Isolate columns for a clean visualization comparison
plot_df = test_data[['Hour', 'Temperature (°F)', 'Predicted Temperature']].copy()

# Melt the DataFrame structure so Seaborn can easily plot "Actual" vs "Predicted" lines
plot_melted = pd.melt(
    plot_df,
    id_vars=['Hour'],
    value_vars=['Temperature (°F)', 'Predicted Temperature'],
    var_name='Data Type',
    value_name='Temperature'
)

# 6. VISUALIZE ACTUAL VS PREDICTED TRENDS
plt.figure(figsize=(12, 6))
sns.set_theme(style="darkgrid")

# Set window title cleanly to bypass the default "Figure 1" title layout
plt.gcf().canvas.manager.set_window_title('Austin 12-Hour Machine Learning Forecast')

line_plot = sns.lineplot(
    data=plot_melted,
    x='Hour',
    y='Temperature',
    hue='Data Type',
    palette={'Temperature (°F)': 'purple', 'Predicted Temperature': 'orange'},
    linewidth=2.5,
    marker='o',
    markersize=8,
    sort=False
)

plt.title(
    "Scikit-Learn 12-Hour Weather Prediction Model\nActual 2026 Forecast vs. Machine Learning Predictions (Austin, TX)",
    fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Hour of the Day", fontsize=11)
plt.ylabel("Temperature (°F)", fontsize=11)
plt.xticks(rotation=15)
plt.legend(frameon=True, facecolor="white")

plt.tight_layout()
plt.show()
