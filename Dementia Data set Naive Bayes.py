#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

 #Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df = pd.read_csv(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv")
df.head()


# In[54]:


df.describe()
df.info()
df.isnull().sum()


# In[55]:


df=df.drop('Prescription',axis=1)
df=df.drop('Dosage in mg', axis=1)


# In[ ]:


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



# In[56]:


df.head()


# In[57]:


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


# In[58]:


##Calculating prior probability of the classes
#m = X_train_scaled.shape[0]
#label_counts = X_train_scaled['class'].value_counts()
#label_counts = np.bincount(y_train_reshaped.flatten())
#prior = np.array(label_counts)/m

# Calculate prior probability of the classes
#label_counts = np.bincount(y_train_reshaped)
#prior_y = label_counts / len(y_train_reshaped)

# calculating Mean and Variance of Features grouped by class labels

#mean_values = x_train.groupby('class')[['sepal length', 'sepal width','petal length','petal width']].mean()
#var_values = x_train.groupby('class')[['sepal length', 'sepal width','petal length','petal width']].var()
'''
# Convert X_train_scaled to a pandas DataFrame
df_train = pd.DataFrame(X_train_scaled, columns=df.columns+1)

# Calculate mean values for each class
mean_values = df_train.groupby('class').mean()

# Calculate variance values for each class
var_values = df_train.groupby('class').var()

# Conveting Pandas Datafram into 2-D Numpy Matrix
means = np.asarray(mean_values)
vars = np.asarray(var_values)
'''


# Calculate mean values for each class
#mean_values = df_train.groupby('class').mean()

# Calculate variance values for each class
#var_values = df_train.groupby('class').var()

# Convert X_train_scaled to a pandas DataFrame
#df_train = pd.DataFrame(X_train_scaled, columns=df.columns)

# Convert X_train_scaled to a pandas DataFrame
df_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Calculate mean values for each class
means = df_train.groupby(y_train_reshaped.flatten()).mean()

# Calculate variance values for each class
var = df_train.groupby(y_train_reshaped.flatten()).var()
#print(means)
#print(var)

#prior = (df_train.groupby(y_train_reshaped.flatten()).count() / len(df_train)).iloc[:,1]# Estimate prior probabilities
# Estimate prior probabilities
prior = (df_train.groupby(y_train_reshaped.flatten()).count() / len(df_train)).iloc[:, 1]

# Storing all possible classes
classes = np.unique(y_train_reshaped.flatten())

#classes = np.unique(df_train[y_train_reshaped.flatten()].tolist()) # Storing all possible classes

'''
###gaussian probability function
def gaussian_probability(x,mu, sigma2):
    a = (1/np.sqrt(2*(np.pi)*sigma2))
    b = np.exp(-np.square(x-mu)/(2*sigma2))
    return a*b

###prediction function for posterior probability
def predict(test_data):
    """
    test_data ndarray(4,)
    """
  # PX_y = P(X|y)
  # Py_X = P(y|X) 
    PX_y = gaussian_probability(test_data,mean_values,var_values) #step -1 
    PX_y = np.prod(PX_y,axis=1) #step - 2
    Py_X = PX_y*prior_y #step - 3
    return np.argmax(Py_X)

'''


# In[59]:


'''
def Normal(n, mu, var):
    
    # Function to return pdf of Normal(mu, var) evaluated at x
    sd = np.sqrt(var)
    pdf = (np.e ** (-0.5 * ((n - mu)/sd) ** 2)) / (sd * np.sqrt(2 * np.pi))
    return pdf
'''
def Normal(n, mu, var):
    sd = np.sqrt(var)
    pdf = np.zeros_like(n)
    mask = sd != 0  # Avoid division by zero
    pdf[mask] = (np.e ** (-0.5 * ((n[mask] - mu) / sd[mask]) ** 2)) / (sd[mask] * np.sqrt(2 * np.pi))
    return pdf


def Predict(X):
    Predictions = []
    
    #for i in X.index:        # Loop through each instances
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
                print(mean,variance,data)
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


# In[60]:


PredictTrain = Predict(X_train_scaled)
PredictTest = Predict(X_test_scaled)

print(PredictTest)


# In[61]:


import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
y_predicted_cls=PredictTest
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


# In[35]:


'''
def Accuracy(y, prediction):
    
    # Function to calculate accuracy
    y = list(y)
    prediction = list(prediction)
    score = 0
    
    for i, j in zip(y, prediction):
        if i == j:
            score += 1
            
    return score / len(y)
    '''


# In[ ]:




