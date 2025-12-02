df = read.csv("kaggle_train.csv")

df$risk_level <- factor(df$risk_level, levels = c(1,2,3,4),labels=c("Low","Medium","High","Very High"))

set.seed(948) # changes seed every test: test1=123, test2=429, test3=948

#seperates 80/20 split; 80% train, 20% test
sample_index <- sample(1:nrow(df), size = 0.8 * nrow(df))
logistic_train_data <- df[sample_index, ]
logistic_test_data <- df[-sample_index, ]

#trains the logistic model
logisticModel <- multinom( risk_level ~ visibility + heart_rate + current_weather, family="multinomial", data = logistic_train_data)

#predicting on the test data
logisticOutcomeProb = predict(logisticModel, newdata = logistic_test_data, type ="probs") #gives probabilty of ea. risk level ocuuring
logisticOutcomeClass = predict(logisticModel, newdata = logistic_test_data, type ="class") #predicts the risk level

#confusion matrix
confusion_matrix = table(Predicted = logisticOutcomeClass, Actual = logistic_test_data$risk_level)
confusionMatrix(confusion_matrix) #function from caret