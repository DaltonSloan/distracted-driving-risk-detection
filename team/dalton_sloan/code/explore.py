from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

df1 = pd.read_csv(DATA_DIR / "kaggle_test.csv")
df2 = pd.read_csv(DATA_DIR / "kaggle_train.csv")



print(df1.describe())
