Background:   Dementia is a prevalent and debilitating condition globally, impacting millions of individuals. 
Understanding the factors associated with its development and progression is crucial for effective management and potential prevention. 
This project seeks to leverage a comprehensive dataset comprising health-related parameters and lifestyle factors to explore their correlation with dementia status.

Dataset Description: The dataset encompasses a wide array of health-related parameters and lifestyle factors. The data includes information on alcohol level, heart rate, blood oxygen level, body temperature, weight,
MRI delay, prescription details, dosage in milligrams, age, education level, dominant hand, gender, family history, smoking status, APOE_ε4 status, physical activity, depression status, cognitive test scores,
medication history, nutrition diet, sleep quality, chronic health conditions, and dementia status. Each row represents a unique individual, and the dataset captures a diverse range of attributes, offering
insights into the correlation between health indicators, lifestyle choices, and medical conditions.

Objective:
● Investigate the relationship between health indicators, lifestyle choices, and dementia status.
● Identify potential risk factors or protective factors associated with dementia.
● Explore patterns related to dementia and associated factors.

Methodology:
● Data Cleaning and Preprocessing: Handle missing values, outliers, and ensure data consistency.
● Exploratory Data Analysis (EDA): Understand the distributions, correlations, and patterns in the
data through summary statistics and visualizations.
● Feature Selection/Engineering: Determine relevant features and create new features if
necessary.
● Modeling and Prediction: Utilize machine learning algorithms Naive Bayes Algorithm to predict dementia status based
on available features. Evaluate model performance using appropriate metrics and techniques.Comparison with a Logistic Regression Model.
● Correlation Analysis: Investigate correlations between dementia status and other variables to
identify significant relationships.

Results:
 with Naive Bayes Classification Algorithm:
 Accuracy: 0.93
Precision: 1.0
Recall: 0.8679245283018868
F1 Score: 0.9292929292929293
ROC AUC Score: 0.9339622641509434
Confusion Matrix:
 [[94  0]
 [14 92]]
with Logistic Regression Sigmoid Function:
Accuracy: 0.485
Precision: 0.5096774193548387
Recall: 0.7452830188679245
F1 Score: 0.6053639846743295
ROC AUC Score: 0.4683861902850261
Confusion Matrix:
 [[18 76]
 [27 79]]
with Logistic Regression Softmax Function:
Accuracy: 0.52
Precision: 0.52
Recall: 0.52
F1 Score: 0.52
ROC AUC Score: 0.49959855479727017
Confusion Matrix:
 [[15 79]
 [17 89]]
