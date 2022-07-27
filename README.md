# Identifiyng and solving Concept Drift detection with two approaches (Adversarial Validation and Kolmogorov-Smirnov test)

# What is concept drift?

Data can change over time. This can result in poor and degrading predictive performance in predictive models that assume a static relationship between input and output variables. This problem of the changing underlying relationships in the data is called concept drift in the field of machine learning.

The term “concept drift” is related to, but distinct from “data drift”. The shift in the relationships between input and output data in the underlying issue over time is called concept drift in machine learning.

Concept drift in machine learning and data mining refers to the change in the relationships between input and output data in the underlying problem over time.

The problem with training examples being different from test examples is that validation won’t be any good for comparing models. Because, during machine learning model development, the model performance on the validation dataset is used as a proxy of the performance on test data. However, if the feature distributions in the training and test datasets are different, the performance on the validation and test datasets will be different. So our cross-validation score is almost unreliable and you shouldn’t use the usual validation techniques. 


This process, can cause concept drift as the distribution of these derived features are could shift over times. This issue is inherent to the data collection and modeling process.


# About project

In the project, I have detected concept drift by using adversarial validation and Kolmogorov-Smirnov test which can also be used in the deployed system.

As a dataset, I have used "Sberbank Russian Housing Market" dataset. Purpose of the problem is to predict the sale price of each property. The target variable is called price_doc. The training data is from August 2011 to June 2015, and the test set is from July 2015 to May 2016. The dataset also includes information about overall conditions in Russia's economy and finance sector.

Firstly, I was trying to detect concept drift so I have used two approaches:

## 1) Concept drift detection via Adversarial Validation

![alt text](https://user-images.githubusercontent.com/33191285/80440265-5e87e580-8943-11ea-996e-ebec215b3dbf.png)



With this approach, the system detects concept drift in new data before making inference, trains a model, and produces predictions adapted to the new data.
The idea is simple: take your training data, remove the target, assemble your training data together with your test data, and create a new binary classification target where the positive label is assigned to the test data.

Then, run a machine learning classifier and evaluate for the ROC-AUC evaluation metric on StratifiedKFold results. If your ROC-AUC is around 0.5, it means that the training and test data are not easily distinguishable and are apparently from the same distribution.

ROC-AUC values higher than 0.5 and nearing 1.0 signal that it is easy for the algorithm to figure out what is from the training set and what is from the test set: in such a case, don’t expect to be able to easily generalize to the test set because it clearly comes from a different distribution.

__In my case, roc_auc_score is above 0.95 which means that model easily distinguish train and test sets using adversarial validation (in other words, we have concept drift problem).__

## 2) Concept drift detection via Kolmogorov-Smirnov test


![alt text](https://miro.medium.com/max/934/0*_zFg_-LPurj7FbPL.)

The main goal of concept drift detection is to determine if two distributions are different. Therefore, the first and most basic approach to infer concept drift applies a hypothesis test to flag if a statistically significant change has occurred between the reference and detection windows for each feature in a given data stream.

If the p-value is less than .05, we reject the null hypothesis. We have sufficient evidence to say that the two sample datasets do not come from the same distribution and it can cause concept drift.

__In my case, 36 features successfully passed Kolmogorov-Smirnov test. We can reject null hypothesis that those features in train and test sets came from the same distribution. Hypothesis that samples are drawn from the same distribution can rejected for 254 features out of 290 based on KS-Test. Which means, we can be worried about concept drift problem.__


## How to solve concept drift problem?

To overcome concept drift problem, I have implemented two stratagies (suppression and constructing custom validaton set that the empirical distribution of the features data is similar to the test data). 

### Method 1: Automated Feature Selection

As I mentioned before, ROC-AUC scores of 0.8 or more would alert you that the test set is peculiar and quite distinguishable from the training data.

If distributions of the features from the train and test data are similar, we expect the adversarial classifier to be as good as random guesses. However, if the adversarial classifier can distinguish between training and test data well (i.e. AUC score ≫ 50%), the top features from the adversarial classifier are potential candidates exhibiting concept drift between the train and test data. We can then exclude these features from model training.

Such feature selection can be automated by determining the number of features to exclude based on the performance of adversarial classifier (e.g. AUC score) and raw feature importance values (e.g. mean decrease impurity (MDI) in Decision Trees) as follows:

(1) Train an adversarial classifier that predicts P({train,test }|X) to separate train and test.

(2) If the AUC score of the adversarial classifier is greater than an AUC threshold θauc , remove features ranked within top x% of remaining features in feature importance ranking and with raw feature importance values higher than a threshold θimp.

(3) Go back to Step 1, if AUC score greater than θauc .

(4) Once the adversarial AUC drops lower than θauc , train an outcome classifier with the selected features and original target variable.

This method is also called suppression.

The only problem with this method is that you may actually be forced to remove the majority of important variables from your data, and any model you then build on such variable censored data won’t be able to predict sufficiently correctly due to the lack of informative features. __In my case, the roc auc score decreased after deleting features with high score.__

## Method 2: Validation Data Selection using two-sample Kolmogorov-Smirnov (KS) test

Finally, with the strategy of validating by mimicking the test set, you keep on training on all the data, but for validation purposes, you pick your examples only from train dataset which have similiar distribution.

photo

For this reason, we will randomly select 250 samples from train dataset in each iteration and use two-sample Kolmogorov-Smirnov (KS) test, which is a non-parametric hypothesis test used to check whether validation candidate data and original test data originate from the same distribution.

If the number of non-rejected features more than rejected features at least three times, we accept the null hypothesis. In this case, we can say that the two sample datasets come from the same distribution. Then we combine these samples and we construct a new validation dataset, so by selecting from the training data so that the empirical distribution of the features data is similar to the test data.

This way, model evaluation metrics on the validation set should get similar results on the test set, which means if the model works well on the validation data, it should work well on the test data.


## Conclusion

From adversarial validation we have evidence that train and test sets come from different distributions. AUC around 0.99 states that XGBoost can easily distinguish train observations from test. These datasets are quite different.

From Kolmogorov-Smirnov Test we can also state that both sets are quite different. Hypothesis that samples are drawn from the same distribution can rejected for 254 features out of 290 based on KS-Test.

After proving concept drift we tried to solve the problem by two strategies: suppression and construct custom validaton set that the empirical distribution of the features data is similar to the test data. 


