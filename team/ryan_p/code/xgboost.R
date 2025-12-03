library(xgboost)
library(Matrix)
library(dplyr)
library(caTools)

df <- read.csv("kaggle_train.csv")

df$speed_violation <- df$speed / df$design_speed
df$heart_rate_dev <- df$heart_rate - mean(df$heart_rate, na.rm = TRUE)
df$env_risk <- df$current_weather * df$precipitation * (1 / df$visibility)
df$hist_weather_risk <- df$accidents_onsite * df$precipitation
df$gear_factor <- df$rpm / (df$speed + 1)
df$engine_stress <- df$engine_load_value * df$engine_temperature

# Remove text columns
df$label_source <- NULL

# Prepare Target (0-3 scale)
labels <- df$risk_level - 1

# Prepare Matrix
data_matrix <- model.matrix(~ . - risk_level - 1, data = df)

# Split data
sample <- sample.split(labels, SplitRatio = 0.80)

dtrain <- xgb.DMatrix(data = data_matrix[sample == TRUE, ], label = labels[sample == TRUE])
dtest  <- xgb.DMatrix(data = data_matrix[sample == FALSE, ], label = labels[sample == FALSE])

# Train model
params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  num_class = 4, 
  eta = 0.03,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

model_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 5000,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 75,
  print_every_n = 20
)

# Evaluate & Visualize

# Predict
pred_probs <- predict(model_xgb, dtest, reshape = TRUE)
pred_labels <- max.col(pred_probs) - 1
actual_labels <- getinfo(dtest, "label")

# Accuracy
accuracy <- sum(pred_labels == actual_labels) / length(actual_labels)
print(paste("Accuracy", round(accuracy * 100, 2), "%"))
