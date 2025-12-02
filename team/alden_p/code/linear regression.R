library(tidyverse)
library(caret)

df <- read.csv("C:/Users/alden/Downloads/kaggle_engineered.csv")

df$risk_level <- as.factor(df$risk_level)

set.seed(789)
idx <- createDataPartition(df$risk_level, p = 0.8, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

model <- lm(as.numeric(risk_level) ~ ., data = train)

pred_num <- predict(model, test)

pred_class <- round(pred_num)

pred_class[pred_class < min(as.numeric(df$risk_level))] <- min(as.numeric(df$risk_level))
pred_class[pred_class > max(as.numeric(df$risk_level))] <- max(as.numeric(df$risk_level))

pred_class <- factor(pred_class, levels = levels(df$risk_level))

truth <- test$risk_level

conf <- confusionMatrix(pred_class, truth)
conf


cm <- confusionMatrix(predictions, test$risk_level)

accuracy <- cm$overall['Accuracy']
recall   <- cm$byClass[,'Sensitivity']
f1       <- cm$byClass[,'F1']

print(accuracy)
print(recall)
print(f1)
