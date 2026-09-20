import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mplcursors
import warnings



# Mute the cursor mismatch warnings at the very start
warnings.filterwarnings("ignore", category=UserWarning, module="mplcursors")

print("📖 Reading localized weather files into Pandas DataFrame...")

try:
    # Load dataset
    master_df = pd.read_csv('austin_weather_history.csv')
except FileNotFoundError:
    print("Error: 'austin_weather_history.csv' not found! Make sure to run download_data.py first.")
    exit()

# Build the multi-line Seaborn plot layout
plt.figure(figsize=(12, 6))
sns.set_theme(style="darkgrid")

# Seaborn assigns the true year names to the line paths
line_plot = sns.lineplot(
    data=master_df,
    x='Hour',
    y='Temperature (°F)',
    hue='Year',
    palette='Set1',
    linewidth=2.5,
    marker='o',
    markersize=8
)

# Connect interactive mouse hover functionality inspired by meteo.com graph
cursor = mplcursors.cursor(line_plot, hover=True)


@cursor.connect("add")
def on_add(sel):
    # 1. Get the precise temperature and hour index under the mouse cursor
    temperature = sel.target[1]
    hour_index = int(round(sel.index))

    unique_hours = master_df['Hour'].unique()

    if 0 <= hour_index < len(unique_hours):
        hour_label = unique_hours[hour_index]

        # 2. Filter data to just this specific hour (e.g., all '04 AM' rows)
        hour_rows = master_df[master_df['Hour'] == hour_label]

        # 3. Find the row in that hour chunk where the temperature matches closest
        # This guarantees we extract the exact year your cursor is hovering near!
        closest_row = hour_rows.iloc[(hour_rows['Temperature (°F)'] - temperature).abs().argsort()[:1]]

        year_label = closest_row['Year'].values[0]
        actual_temp = closest_row['Temperature (°F)'].values[0]

        # 4. Display the perfect string match layout
        sel.annotation.set_text(f"{hour_label} | Year {year_label}\nTemp: {actual_temp:.1f}°F")
    else:
        sel.annotation.set_text(f"{sel.target[1]:.1f}°F")

    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, boxstyle="round,pad=0.5")


# Visual formatting rules
plt.title("Year-over-Year 12-Hour Weather Trend in Austin, United States\nSaved Local Data for September 15",
          fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Temperature (°F)", fontsize=12)
plt.xticks(rotation=15)
plt.legend(title="Comparison Years", frameon=True, facecolor="white")
plt.gcf().canvas.manager.set_window_title('Austin Year-over-Year Weather Analysis')
plt.tight_layout()
plt.show()
