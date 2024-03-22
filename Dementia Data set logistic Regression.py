#!/usr/bin/env python
# coding: utf-8

# In[317]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

 #Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[318]:


df = pd.read_csv(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv")
df.head()


# In[319]:


df.describe()


# In[320]:


df.info()


# In[321]:


df.isnull().sum()


# In[322]:


df=df.drop('Prescription',axis=1)
df=df.drop('Dosage in mg', axis=1)


# In[323]:


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



# In[324]:


df.head()


# In[325]:


##Feature Scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


# In[326]:


##Define the hyperparameters
learning_rate=0.01
epochs=1000

##Initializing weights and bias
weight= np.random.rand(21,1) 
bias=np.random.rand()
#print(weight,bias)
##sigmoid function
def sigmoid(z):
    # Convert z to a NumPy array or matrix if it's not already
    #z = np.array(z)
    return 1/(1+np.exp(-z))

##training loop
for epoch in range(epochs):
    ##forward pass
    z=weight.T*X_train_scaled + bias
    #print(z)
    predicted= sigmoid(z)
    #print(predicted)
    
    ##loss computing
    epsilon = 1e-15  # Small value to avoid log(0) issues

# Clip predicted values to avoid log(0) and log(1) issues
    predicted = np.clip(predicted, epsilon, 1 - epsilon)

    #calculation ofloss
    loss=-(1/1000)*np.sum(y_train_reshaped*np.log(predicted)+(1-y_train_reshaped)*np.log(1-predicted))
    #print(loss)
    ##gradient computing
    d_wt= (1/1000)*np.sum((predicted-y_train_reshaped)*X_train_scaled)
    d_bias=(1/1000)*np.sum(predicted-y_train_reshaped)
    ##update the weights and bias
    weight= weight - learning_rate*d_wt
    bias= bias - learning_rate*d_bias
    
    if epoch%100 ==0:
        print(f'Epoch(epoch),Loss(loss)')


# In[327]:


##testing the data after training
new_x= X_test_scaled
z= weight.T*new_x + bias
predicted = sigmoid(z)
### to find the index (or class) with the highest predicted probability for each sample in the dataset.
highest_probability_feature_index=np.argmax(predicted,axis=1)
y_predicted = np.max(predicted, axis=1)

print(predicted)
print(y_predicted)


# In[329]:


# Assuming y_predicted contains continuous values
threshold = 0.6 # Adjust this threshold as needed

# Flatten y_predicted if it's a multi-dimensional array
#flattened_y_predicted = predicted.flatten()

# Threshold the flattened predictions
y_predicted_cls = [1 if i > 0.59 else 0 for i in y_predicted]

#y_predicted_binary = (y_predicted > threshold).astype(int)

#y_predicted_cls = [1 if i > 0.5 else 0 for i in predicted]
#y_pred=np.array(y_predicted_cls)
print(y_predicted_cls)

print(y_test_reshaped)


# In[330]:


import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

###evaluation of the model
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[255]:





# In[ ]:




