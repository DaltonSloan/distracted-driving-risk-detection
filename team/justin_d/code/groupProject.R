# Load dataset

kaggle_train <- read.csv("C:\Users\jdiet\OneDrive\Desktop\kaggle_train.csv")

library(dplyr)
library(rpart)
library(tidyverse)
library(tidymodels)
library(tidymodels, lib.loc = tempdir())


# Standardize all selected features
features <- c("speed","rpm","acceleration","throttle_position",
              "engine_temperature","engine_load_value","heart_rate",
              "observation_hour","current_weather","visibility",
              "precipitation","accidents_onsite","design_speed",
              "accidents_time")

df <- kaggle_train
df[features] <- scale(kaggle_train[features])


# EDA, discover which features had the greatest impact on the different risk levels
head(kaggle_train)

features <- c("speed","rpm","acceleration","throttle_position",
              "engine_temperature","engine_load_value","heart_rate",
              "observation_hour","current_weather","visibility",
              "precipitation","accidents_onsite","design_speed",
              "accidents_time")

feature_means <- aggregate(. ~ risk_level, data=kaggle_train[, c("risk_level", features)], mean)


# Remove risk_level column
mean_values <- feature_means[, features]


# Compute the difference between max and min mean for each feature
mean_change <- apply(mean_values, 2, function(x) {max(x) - min(x)})
print(sort(mean_change, decreasing = TRUE))


# Engineered Features
df$speed_violation <- df$speed / df$design_speed
df$heart_rate_dev <- df$heart_rate - mean(df$heart_rate)
df$env_risk <- df$current_weather * df$precipitation * (1/df$visibility)


#linear regression
model_lm <- lm(risk_level ~ speed_violation + heart_rate_dev + env_risk, data = df)
summary(model_lm)


#kNN
library(class)


# Prepare features
features <- c("speed_violation","heart_rate_dev","env_risk")


# For demonstration, we’ll predict on the same training set
predicted_knn <- knn(train = df[features], test = df[features], cl = df$risk_level, k = 5)


# Compare
table(predicted_knn, df$risk_level)
mean(predicted_knn == df$risk_level)  # Accuracy


# Naive Bayes
install.packages("e1071")
install.packages("caret")
library(e1071)
library(lattice)
library(caret)

set.seed(123)  # for reproducibility

# Split training data 80/20
n <- nrow(kaggle_train)
train_idx <- sample(1:n, size = 0.8*n)
train_split <- kaggle_train[train_idx, ]
valid_split <- kaggle_train[-train_idx, ]


# Standardize train and valid splits
num_cols <- sapply(train_split, is.numeric)
num_cols["risk_level"] <- FALSE

train_means <- apply(train_split[, num_cols], 2, mean)
train_sds   <- apply(train_split[, num_cols], 2, sd)

train_split[, num_cols] <- scale(train_split[, num_cols], center = train_means, scale = train_sds)
valid_split[, num_cols] <- scale(valid_split[, num_cols], center = train_means, scale = train_sds)


# Naive Bayes with all features
nb_all <- naiveBayes(risk_level ~ ., data = train_split, laplace = 1)
pred_all <- predict(nb_all, valid_split)
accuracy_all <- mean(pred_all == valid_split$risk_level)
accuracy_all

# Confusion Matrix
nb_cm <- table(Predicted = pred_all, Actual = valid_split$risk_level)
nb_cm


# Calculate engineered features for Naive Bayes
train_split$speed_violation <- train_split$speed / train_split$design_speed
valid_split$speed_violation <- valid_split$speed / valid_split$design_speed

train_split$heart_rate_dev <- train_split$heart_rate - mean(train_split$heart_rate)
valid_split$heart_rate_dev <- valid_split$heart_rate - mean(train_split$heart_rate)  

train_split$env_risk <- train_split$current_weather * train_split$precipitation / train_split$visibility
valid_split$env_risk <- valid_split$current_weather * valid_split$precipitation / valid_split$visibility


# Naive Bayes with engineered features
engineered_features <- c("speed_violation", "heart_rate_dev", "env_risk")
train_eng <- train_split[, c(engineered_features, "risk_level")]
valid_eng <- valid_split[, engineered_features]

nb_eng <- naiveBayes(risk_level ~ ., data = train_eng, laplace = 1)
pred_eng <- predict(nb_eng, valid_eng)
accuracy_eng <- mean(pred_eng == valid_split$risk_level)
accuracy_eng

# Confusion Matrix
nb_eng_cm <- table(Predicted = pred_eng, Actual = valid_split$risk_level)
nb_eng_cm

