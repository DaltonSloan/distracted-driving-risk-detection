from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Load data
df = pd.read_csv(DATA_DIR / "kaggle_train.csv")

# Stress measure
stress = "heart_rate"

# Variables to compare
predictors = [
    "speed",
    "rpm",
    "visibility",
    "precipitation",
    "accidents_onsite"
]

# Correlation matrix for selected columns
corr = df[predictors + [stress]].corr()

# Extract just predictor→stress correlations
corr_to_stress = corr[[stress]].drop(index=[stress])

# ---- Seaborn Styling ----
sns.set_theme(style="white", context="talk")

plt.figure(figsize=(9, 6))

ax = sns.heatmap(
    corr_to_stress,
    annot=True,
    fmt=".2f",
    linewidths=1.5,
    linecolor="white",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    cbar_kws={"shrink": 0.8, "label": "Correlation Strength"},
    annot_kws={"fontsize": 16, "fontweight": "bold"}
)

# Title & labels
plt.title("Stress Predictors Correlation Map\n(Heart Rate as Stress Indicator)", fontsize=22, pad=20)
plt.xlabel("Heart Rate (Stress)", fontsize=16)
plt.ylabel("Predictor", fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14, rotation=0)

plt.tight_layout()
plt.show()
