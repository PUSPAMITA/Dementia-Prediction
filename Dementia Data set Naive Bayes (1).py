#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split  ##for splitting training and test data
from sklearn.preprocessing import StandardScaler   ##feature scaling
##for assesing the accuracy of our model
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import confusion_matrix
##for plotting figures
import matplotlib.pyplot as plt

 #Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#This code iterates through all files within a directory and its subdirectories, printing the full path of each file.
import os 
for dirname, _, filenames in os.walk(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[72]:


df = pd.read_csv(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv")
df.head()


# In[73]:


df.describe()


# In[68]:


df.info()


# In[74]:


df.isnull().sum()


# In[58]:


###dropping features since there is more than 515 blanks out of 1000.
df=df.drop('Prescription',axis=1)
df=df.drop('Dosage in mg', axis=1)


# In[59]:


### Replace non-numerical values with integers in columns
### Define a function to replace non-numerical values with integers
def replace_education_level(x):
    if x == 'Primary School':
        return 1
    elif x == 'Secondary School':
        return 2
    elif x == 'Diploma/Degree':
        return 3
    else:
        return 0

# Replace non-numerical values with integers in column 'Education_Level'
df['Education_Level'] = df['Education_Level'].apply(replace_education_level)

### Define a function to replace non-numerical values with integers
def replace_dominant_hand(x):
    if x == 'Right':
        return 1
    else:
        return 2
# Replace non-numerical values with integers in column 'Dominant Hand'
df['Dominant_Hand'] = df['Dominant_Hand'].apply(replace_dominant_hand)

### Define a function to replace non-numerical values with integers
def replace_gender(x):
    if x == 'Female':
        return 1
    else:
        return 2
# Replace non-numerical values with integers in column 'Gender'
df['Gender'] = df['Gender'].apply(replace_gender)

### Define a function to replace non-numerical values with integers
def replace_family_history(x):
    if x == 'Yes':
        return 1
    else:
        return 0
# Replace non-numerical values with integers in column 'Family_History'
df['Family_History'] = df['Family_History'].apply(replace_family_history)

### Define a function to replace non-numerical values with integers
def replace_smoking_history(x):
    if x == 'Former Smoker':
        return 1
    elif x == 'Current Smoker':
        return 2
    else:
        return 0
# Replace non-numerical values with integers in column 'Smoking_Status'
df['Smoking_Status'] = df['Smoking_Status'].apply(replace_smoking_history)

### Define a function to replace non-numerical values with integers
def replace_APOE_ε4(x):
    if x == 'Positive':
        return 1
    else:
        return 0
# Replace non-numerical values with integers in column 'APOE_ε4'
df['APOE_ε4'] = df['APOE_ε4'].apply(replace_APOE_ε4)


### Define a function to replace non-numerical values with integers
def replace_Physical_Activity(x):
    if x == 'Mild Activity':
        return 1
    elif x == 'Moderate Activity':
        return 2
    else:
        return 0
# Replace non-numerical values with integers in column 'Physical_Activity'
df['Physical_Activity'] = df['Physical_Activity'].apply(replace_Physical_Activity)    


### Define a function to replace non-numerical values with integers
def replace_Depression_Status(x):
    if x == 'Yes':
        return 1
    else:
        return 0
# Replace non-numerical values with integers in column 'Depression_Status'
df['Depression_Status'] = df['Depression_Status'].apply(replace_Depression_Status)

### Define a function to replace non-numerical values with integers
def replace_Medication_History(x):
    if x == 'Yes':
        return 1
    else:
        return 0
# Replace non-numerical values with integers in column 'Medication_History'
df['Medication_History'] = df['Medication_History'].apply(replace_Medication_History)

### Define a function to replace non-numerical values with integers
def replace_Nutrition_Diet(x):
    if x == 'Low-Carb Diet':
        return 1
    elif x == 'Mediterranean Diet':
        return 2
    else:
        return 0
# Replace non-numerical values with integers in column 'Nutrition_Diet'
df['Nutrition_Diet'] = df['Nutrition_Diet'].apply(replace_Nutrition_Diet)    

### Define a function to replace non-numerical values with integers
def replace_Sleep_Quality(x):
    if x == 'Poor':
        return 1
    else:
        return 0
# Replace non-numerical values with integers in column 'Sleep_Quality'
df['Sleep_Quality'] = df['Sleep_Quality'].apply(replace_Sleep_Quality)

### Define a function to replace non-numerical values with integers
def replace_Chronic_Health_Conditions(x):
    if x == 'Heart Disease':
        return 1
    elif x == 'Hypertension':
        return 2
    elif x == 'Diabetes':
        return 3
    else:
        return 0
# Replace non-numerical values with integers in column 'Chronic_Health_Conditions'
df['Chronic_Health_Conditions'] = df['Chronic_Health_Conditions'].apply(replace_Chronic_Health_Conditions)



# In[60]:


df.head()


# In[61]:


##Feature Scaling


X = df.drop('Dementia', axis=1)
y = df['Dementia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# Convert Pandas Series to NumPy array
y_train_array = np.array(y_train)

# Reshape the array
y_train_reshaped = y_train_array.reshape((800, 1))

# Convert Pandas Series to NumPy array
y_test_array = np.array(y_test)

# Reshape the array
y_test_reshaped = y_test_array.reshape((200, 1))


# In[77]:


##Naive Bayes algorithm

# Convert X_train_scaled to a pandas DataFrame
df_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# Calculate mean values for each class
means = df_train.groupby(y_train_reshaped.flatten()).mean()
# Calculate variance values for each class
var = df_train.groupby(y_train_reshaped.flatten()).var()
#print(means)
#print(var)
# Calculation of prior probabilities in each class
prior = (df_train.groupby(y_train_reshaped.flatten()).count() / len(df_train)).iloc[:, 1]  # Estimate prior probabilities
#print(prior)
# Storing all possible classes
classes = np.unique(y_train_reshaped.flatten())
#print(classes)


# In[80]:


def Normal(n, mu, var):
    sd = np.sqrt(var)
    pdf = np.zeros_like(n)
    mask = sd != 0  # Avoid division by zero
    pdf[mask] = (np.e ** (-0.5 * ((n[mask] - mu) / sd[mask]) ** 2)) / (sd[mask] * np.sqrt(2 * np.pi))
    return pdf

def Predict(X):
    Predictions = []
    # Loop through each instances
    for instance in X:   
        ClassLikelihood = []
        #instance = X.loc[i]
        for cls in classes: # Loop through each class
            FeatureLikelihoods = []
            FeatureLikelihoods.append(np.log(prior[cls])) # Append log prior of class 'cls'
            for col in range (X.shape[1]): # Loop through each feature
                #data = instance[col]
                mean = means.iloc[cls,col] # Find the mean of column 'col' that are in class 'cls'
                variance = var.iloc[cls,col] # Find the variance of column 'col' that are in class 'cls'
                data = instance[col]
                #print(mean,variance,data)
                Likelihood = Normal(data, mean, variance)
                if Likelihood != 0:
                    Likelihood = np.log(Likelihood) # Find the log-likelihood evaluated at x
                else:
                    Likelihood = 1/len(df_train) 
                FeatureLikelihoods.append(Likelihood)
            TotalLikelihood = sum(FeatureLikelihoods) # Calculate posterior
            ClassLikelihood.append(TotalLikelihood) 
        MaxIndex = ClassLikelihood.index(max(ClassLikelihood)) # Find largest posterior position
        Prediction = classes[MaxIndex]
        Predictions.append(Prediction) 
    return Predictions


# In[86]:


PredictTrain = Predict(X_train_scaled)
PredictTest = Predict(X_test_scaled)
#print(PredictTrain)
print(PredictTest)


# In[82]:


###evaluation of the model
y_predicted_cls=PredictTest

accuracy = accuracy_score(y_test_reshaped, y_predicted_cls)
precision = precision_score(y_test_reshaped, y_predicted_cls)
recall = recall_score(y_test_reshaped, y_predicted_cls)
f1 = f1_score(y_test_reshaped, y_predicted_cls)
roc_auc = roc_auc_score(y_test_reshaped, y_predicted_cls)
conf_matrix = confusion_matrix(y_test_reshaped, y_predicted_cls)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.metrics import roc_curve, roc_auc_score
# Assuming you have true labels and predicted class probabilities
fpr, tpr, thresholds = roc_curve(y_test_reshaped, y_predicted_cls)


# In[87]:


#PLOTTING THE STATISTICAL MEASURES.
# Create subplots for plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16, 4))
# Plot accuracy, precision, recall, F1-score
ax1.bar(['Accuracy', 'Precision', 'Recall', 'F1-Score'], [accuracy, precision, recall, f1])
ax1.set_xlabel('Metrics')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Metrics')
# Plot confusion matrix as heatmap
ax2.imshow(conf_matrix, cmap='Blues')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
ax2.set_title('Confusion Matrix')
ax2.text(0, 0, str(conf_matrix[0, 0]), ha='center', va='center', fontsize=12, color='white')
ax2.text(0, 1, str(conf_matrix[0, 1]), ha='center', va='center', fontsize=12, color='black')
ax2.text(1, 0, str(conf_matrix[1, 0]), ha='center', va='center', fontsize=12, color='black')
ax2.text(1, 1, str(conf_matrix[1, 1]), ha='center', va='center', fontsize=12, color='white')
# Placeholder for ROC AUC 
ax3.text(0.5, 0.5,  f'ROC AUC\n{roc_auc:.4f}', ha='center', va='center', fontsize=12)
ax3.set_title('ROC AUC')
ax3.axis('off')
#Plot ROC Curve
plt.figure(figsize=(6, 3))
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid(True)
# Add diagonal line for random classification
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
# Adjust layout
plt.tight_layout()
plt.legend()
plt.show()


# In[ ]:




