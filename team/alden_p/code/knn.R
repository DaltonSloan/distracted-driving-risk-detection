library(tidyverse)
library(caret)
library(class)

df <- read.csv("C:/Users/alden/Downloads/kaggle_train_filtered.csv")


df$risk_level <- as.factor(df$risk_level)

set.seed(789)
train_index <- createDataPartition(df$risk_level, p = 0.8, list = FALSE)

train <- df[train_index, ]
test  <- df[-train_index, ]

preproc <- preProcess(train %>% select(-risk_level), method = c("center", "scale"))

train_norm <- predict(preproc, train)
test_norm  <- predict(preproc, test)

x_train <- train_norm %>% select(-risk_level)
y_train <- train_norm$risk_level

x_test <- test_norm %>% select(-risk_level)
y_test <- test_norm$risk_level

k <- 5
predictions <- knn(train = x_train, test = x_test, cl = y_train, k = k)

conf_matrix <- confusionMatrix(predictions, y_test)
print(conf_matrix)

cm <- confusionMatrix(predictions, test$risk_level)

accuracy <- cm$overall['Accuracy']
recall   <- cm$byClass[,'Sensitivity']
f1       <- cm$byClass[,'F1']

print(accuracy)
print(recall)
print(f1)