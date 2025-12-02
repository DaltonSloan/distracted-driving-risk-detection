from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Load your dataset
df = pd.read_csv(DATA_DIR / "kaggle_train.csv")

# Ensure the needed columns are present
required_cols = ["current_weather", "heart_rate"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")

# Group by weather condition and compute average heart rate
weather_hr = df.groupby("current_weather")["heart_rate"].mean()

# Plot
plt.figure(figsize=(10, 6))
weather_hr.plot(kind='bar', color='skyblue')

plt.title("Average Heart Rate Across Weather Conditions")
plt.xlabel("Weather Condition Code")
plt.ylabel("Average Heart Rate (bpm)")
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
