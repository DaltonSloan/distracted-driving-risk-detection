# Run these lines once if you don't have the packages
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("dplyr")
# install.packages("caTools") # For splitting data

# Load the libraries for this session
library(rpart)
library(rpart.plot)
library(dplyr)
library(caTools)


# Load data
df <- read.csv("kaggle_train.csv")

# Engineered Features
df$speed_violation <- df$speed / df$design_speed
df$heart_rate_dev <- df$heart_rate - mean(df$heart_rate) 
df$env_risk <- df$current_weather * df$precipitation * 	(1/df$visibility)
df$hist_weather_risk <- df$accidents_onsite * df$precipitation
df$gear_factor <- df$rpm / (df$speed + 1) 
df$engine_stress <- df$engine_load_value * df$engine_temperature 

# Create a mask to split the data: 80% for training, 20% for testing
sample <- sample.split(df$risk_level, SplitRatio = 0.80)

train_data <- subset(df, sample == TRUE)
test_data  <- subset(df, sample == FALSE)

# We build the model on the training data
# Formula: "risk_level ~ ." means "predict risk_level using ALL other columns"
# method = "class" tells rpart to do classification (not regression)
tree_model <- rpart(
  risk_level ~ ., 
  data = train_data, 
  method = "class"
)

print(tree_model)
rpart.plot(tree_model, main = "Decision Tree for Risk Level")
predictions <- predict(tree_model, test_data, type = "class")

# Create a Confusion Matrix
conf_matrix <- table(Predicted = predictions, Actual = test_data$risk_level)

print("Confusion Matrix:")
print(conf_matrix)

# Sum of diagonal (correct predictions) divided by total predictions
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Overall Accuracy:", round(accuracy, 4)))
