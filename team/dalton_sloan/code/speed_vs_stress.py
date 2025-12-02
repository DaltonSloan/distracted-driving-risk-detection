from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Load data
df = pd.read_csv(DATA_DIR / "kaggle_train.csv")

# Stress = heart_rate
df = df[df["heart_rate"] > 0]  # basic cleaning to avoid weird values

# Set plot theme
sns.set_theme(style="whitegrid", context="talk")

plt.figure(figsize=(12, 8))

# Create a scatterplot + regression line + KDE shading
sns.jointplot(
    data=df,
    x="speed",
    y="heart_rate",
    kind="reg",         # regression plot
    height=10,
    scatter_kws={"alpha": 0.3, "color": "steelblue"},
    line_kws={"color": "darkred"},
)

# Additional styling
plt.suptitle("Relationship Between Vehicle Speed and Driver Stress (Heart Rate)", fontsize=18, y=1.02)

# Show plot
plt.show()
