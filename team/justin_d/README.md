# Justin Dieter

This folder holds my work for the project

## What’s here
- `code/` — my scripts (EDA, Engineered Features, KNN, Naive Bayes).
- This README — summary of what I tried, what worked, and what failed.

## Experiments (What I tried)
- Started by standardizing all of our features in the EDA and tried discovering which features had the greatest impact.
    I did this by calculating the mean of every feature, for each risk level. I then measured how much the means differed between the different levels. Features with the greatest variation are more predictive.

- After I found which features had the greatest impact I created engineered features. I created the features: 

    > speed_violation — how much the driver exceeded the safe or expected speed
    > heart_rate_dev — how far the driver’s heart rate was from their baseline
    > env_risk — combined weather + road conditions into a single risk score

    By doing this, the data became easier to interpret and the models picked up risk patterns more consistently 

- After this I tried creating a KNN for a baseline to see if my engineered features were separating risk levels. This however, wasnt very useful because KNN struggles when features overlap heavily and when they dont form tight clusters.
    In the model medium and high risk blended too much making KNN not reliable at predicting. Thus I desided to go back and try some other models
- I then tried creating a Naive Bayes model. I tried this model using my engineered features and also tried it just using all the features.
    Using just my engineered features the model gave a good baseline of 56% accuracy. It performed well on low and very high-risk categories, but struggled with middle levels.
```
    Confusion Matrix:
           Actual
   Predicted  1  2  3  4
           1 97 31 15  1
           2  5 34 20  2
           3 11 18 10  4
           4  1 17 49 81
```
Interestingly when using all the features in this model we get a accuracy of 60%. This is slightly higher than just using the engineered features. Thus, this suggests that the engineered features help the model, but the full set of features still contained additional useful information for boosting the accuracy.
```
    Confusion Matrix:
           Actual
Predicted  1  2  3  4
        1 90 24  2  0
        2 20 43 24  0
        3  3 14 21  4
        4  1 19 47 84
```

## What worked
- Engineered features helped improve separation between low and very high-risk categories.
- Standardizing the features improved performance across every model and made our results more accurate.
- Naive Bayes performed the best out of the models I tested.

## What failed
- KNN performed poorly because the risk levels dont form clean clusters in feature space.
- Engineered features alone werent enough, they still missed too much variation in the mid-risk categories.
- Linear models werent useful for classification, and didnt capture the non-linear relationships in the data.