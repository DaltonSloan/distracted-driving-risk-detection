install.packages("tidyverse")
install.packages("randomForest")
install.packages("caret")

library(tidyverse)
library(randomForest)
library(caret)

data <- read.csv("C:/Users/alden/Downloads/kaggle_train_filtered.csv")

str(data)
summary(data)

data$risk_level <- as.factor(data$risk_level)

set.seed(789)
trainIndex <- createDataPartition(data$risk_level, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test  <- data[-trainIndex, ]

rf_model <- randomForest(
  risk_level ~ ., 
  data = train,
  ntree = 500,          
  mtry = floor(sqrt(ncol(train) - 1)),  
  importance = TRUE
)

print(rf_model)
varImpPlot(rf_model)

predictions <- predict(rf_model, newdata = test)

conf_matrix <- confusionMatrix(predictions, test$risk_level)
print(conf_matrix)

importance(rf_model)

cm <- confusionMatrix(predictions, test$risk_level)

accuracy <- cm$overall['Accuracy']
recall   <- cm$byClass[,'Sensitivity']
f1       <- cm$byClass[,'F1']

print(accuracy)
print(recall)
print(f1)