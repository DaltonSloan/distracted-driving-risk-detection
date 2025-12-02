install.packages("nnet") # for multinomial regression
install.packages("caret")
library(nnet)
library(caret)

df = read.csv("kaggle_train.csv")


df$risk_level <- factor(df$risk_level, levels = c(1,2,3,4),labels=c("Low","Medium","High","Very High"))

#eda variables
df$speed_violation <- df$speed / df$design_speed 
df$heart_rate_dev <- df$heart_rate - mean(df$heart_rate) 
df$env_risk <- df$current_weather * df$precipitation * 	(1/df$visibility) 

set.seed(123) # changes seed every test: test1=948, test2=429, test3=123

#seperates 80/20 split; 80% train, 20% test
sample_index <- sample(1:nrow(df), size = 0.8 * nrow(df))
logistic_train_data <- df[sample_index, ]
logistic_test_data <- df[-sample_index, ]

#trains the logistic model
logisticModel <- multinom( risk_level ~ speed_violation + heart_rate_dev + env_risk, family="multinomial", data = logistic_train_data)

#predicting on the test data
logisticOutcomeProb = predict(logisticModel, newdata = logistic_test_data, type ="probs") #gives probabilty of ea. risk level ocuuring
logisticOutcomeClass = predict(logisticModel, newdata = logistic_test_data, type ="class") #predicts the risk level
logisticOutcomeProb

#confusion matrix
confusion_matrix = table(Predicted = logisticOutcomeClass, Actual = logistic_test_data$risk_level)
confusionMatrix(confusion_matrix) #function from caret