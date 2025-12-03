install.packages("nnet") # for multinomial regression
install.packages("caret")
library(nnet)
library(caret)

df = read.csv("kaggle_train.csv")


df$risk_level <- factor(df$risk_level, levels = c(1,2,3,4),labels=c("Low","Medium","High","Very High"))

#the engineered variables
df$speed_violation <- df$speed / df$design_speed 
df$heart_rate_dev <- df$heart_rate - mean(df$heart_rate) 
df$env_risk <- df$current_weather * df$precipitation * 	(1/df$visibility) 

set.seed(123) # changes seed every test: test1=948, test2=429, test3=123

#seperates 80/20 split; 80% train, 20% test
sample_index <- sample(1:nrow(df), size = 0.8 * nrow(df))
logistic_train_data <- df[sample_index, ]
logistic_test_data <- df[-sample_index, ]

#trains the logistic model using the engineered variables
logisticModel <- multinom( risk_level ~ speed_violation + heart_rate_dev + env_risk, family="multinomial", data = logistic_train_data)

#trains the logistic model using the given variables
#logisticModel <- multinom( risk_level ~ visibility + heart_rate + current_weather, family="multinomial", data = logistic_train_data)


#predicting on the test data
logisticOutcomeProb = predict(logisticModel, newdata = logistic_test_data, type ="probs") #gives probabilty of ea. risk level ocuuring
logisticOutcomeClass = predict(logisticModel, newdata = logistic_test_data, type ="class") #predicts the risk level
logisticOutcomeProb

#confusion matrix
confusion_matrix = table(Predicted = logisticOutcomeClass, Actual = logistic_test_data$risk_level)
confusionMatrix(confusion_matrix) #function from caret





Test 1 (seed = 123):
 Actual
Predicted Low Medium High Very High
      Low 82 24  2  0
   Medium 30 32 16  0
     High  1 29 32  9
Very High  1 15 44 79
accuracy = 56.8%
      Low risk: Recall = 71.9%, Precision = 75.9%, F1 = 73.9%
   Medium Risk: Recall = 32.0%, Precision = 41.0%, F1 = 35.9%
     High Risk: Recall = 34.0%, Precision = 45.1%, F1 = 38.7%
Very High Risk: Recall = 90.0%, Precision = 56.8%, F1 = 69.6%
Test 2 (seed = 429):
 Actual
Predicted Low Medium High Very High
      Low  81     31    3         0
   Medium  22     33   16         0
     High   0     24   38         7
Very High   0      9   43        89
accuracy = 61.9%
      Low risk: Recall = 70.4%, Precision = 78.6%, F1 = 74.30%
   Medium Risk: Recall = 37.5%, Precision = 41.3%, F1 = 39.3%
     High Risk: Recall = 38.0%, Precision = 55.1%, F1 = 45.0%
Very High Risk: Recall = 92.7%, Precision = 67.4%, F1 = 78.1%
Test 3 (seed = 948):
 Actual 
Predicted  Low Medium High Very High
      Low  73      33    4         0
   Medium  13      48   20         0
     High   3      19   24         6
Very High   0      10   45        98
accuracy = 61.4%
      Low risk: Recall = 66.3%, Precision = 82.0%, F1 = 73.3%
   Medium Risk: Recall = 53.3%, Precision = 47.5%, F1 = 50.2%
     High Risk: Recall = 25.8%, Precision = 46.2,  F1 = 33.0%
Very High Risk: Recall = 94.2%, Precision = 68.5%, F1 = 79.3%