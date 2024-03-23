#!/usr/bin/env python
# coding: utf-8

# In[628]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

 #Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[629]:


df = pd.read_csv(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv")
df.head()


# In[630]:


df.describe()


# In[631]:


df.info()


# In[632]:


df.isnull().sum()


# In[633]:


df=df.drop('Prescription',axis=1)
df=df.drop('Dosage in mg', axis=1)


# In[634]:


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



# In[635]:


df.head()


# In[636]:


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


# In[637]:


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
        print(f'Epoch{epoch},Loss{loss}')


# In[638]:


##testing the data after training
new_x= X_test_scaled
z= weight.T*new_x + bias
predicted = sigmoid(z)
### to find the index (or class) with the highest predicted probability for each sample in the dataset.
highest_probability_feature_index=np.argmax(predicted,axis=1)
y_predicted = np.max(predicted, axis=1)

print(predicted)
print(y_predicted)


# In[639]:


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


# In[640]:


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





# In[641]:


##Applying Logistic Regression Model Using Softmax Function(Multinomial Classification)

def one_hot(y, c):
    
    # y--> label/ground truth.
    # c--> Number of classes.
    
    # A zero matrix of size (m, c)
    y_hot = np.zeros((len(y), c))
    
    # Putting 1 for column where the label is,
    # Using multidimensional indexing.
    y_hot[np.arange(len(y)), y] = 1
    
    return y_hot   


# In[642]:


def softmax(z):
    
    # z--> linear part.
    
    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z))
    
    # Calculating softmax for all examples.
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
        
    return exp


# In[643]:


#loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))


# In[644]:


def fit(X, y, lr, c, epochs):
    
    # X --> Input.
    # y --> true/target value.
    # lr --> Learning rate.
    # c --> Number of classes.
    # epochs --> Number of iterations.
    
        
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Initializing weights and bias randomly.
    w = np.random.random((n, c))
    b = np.random.random(c)    # Empty list to store losses.
    losses = []
    
    # Training loop.
    for epoch in range(epochs):
        
        # Calculating hypothesis/prediction.
        z = X@w + b
        y_hat = softmax(z)
        
        # One-hot encoding y.
        y_hot = one_hot(y, c)
        
        # Calculating the gradient of loss w.r.t w and b.
        w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot)) 
        b_grad = (1/m)*np.sum(y_hat - y_hot)
        
        # Updating the parameters.
        w = w - lr*w_grad
        b = b - lr*b_grad
        
        # Calculating loss and appending it in the list.
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        losses.append(loss)        # Printing out the loss at every 100th iteration.
        if epoch%100==0:
            print('Epoch {epoch}==> Loss = {loss}'
                  .format(epoch=epoch, loss=loss))    
            return w, b, losses


# In[645]:


# Flattening the image.
#X_train = train_X.reshape(60000,28*28)# Normalizing. 
#X_train = X_train/255# Training
w, b, l = fit(X_train_scaled, y_train_reshaped, lr=0.01, c=21, epochs=1000)


# In[651]:


def predict(X, w, b):
    
    # X --> Input.
    # w --> weights.
    # b --> bias.
    
    # Predicting
    z = X@w + b
    y_hat = softmax(z)
    print(y_hat)
    
    # One-hot encoding y.
    #y_hat = one_hot(y, c=21)
    
    # Returning the class with highest probability.
    #return np.argmax(y_hat, axis=1)
    #return np.max(y_hat, axis=1)
    return y_hat


# In[652]:


predict(X_test_scaled,w,b)

#y_hat = [1 if i > 0.4 else 0 for i in predict(X_test_scaled,w,b)]
# One-hot encoding y.
#y_test = one_hot(y_test_reshaped, c=21)


#from sklearn.preprocessing import OneHotEncoder

# Convert y_hat to multilabel-indicator format
#encoder = OneHotEncoder(sparse_output=False)
y_hat_multilabel = one_hot(y_hat, c=21)
#print(y_hat_multilabel)

 #Assuming y_hat_multilabel contains the predicted probabilities for each class
threshold = 0.59  # Threshold for binary classification

# Convert probabilities to binary predictions based on the threshold
y_predicted = np.where(y_hat_multilabel > threshold, 1, 0)
print(y_predicted)
#y_predicted = [1 if i > 0.59 else 0 for i in y_hat_multilabel]
#print(y_test)
#print(y_test_reshaped)
# Now both y_test and y_hat_multilabel should be in multilabel-indicator format


# In[653]:


import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.preprocessing import OneHotEncoder

# Convert y_hat to multilabel-indicator format
encoder = OneHotEncoder(sparse_output=False)
y_predicted = encoder.fit_transform(y_hat_multilabel)
y_test=encoder.fit_transform(y_test_reshaped)

print(y_predicted.shape[0])
print(y_test.shape[0])

# Convert multilabel-indicator format to a binary label format
y_test_labels = np.argmax(y_test, axis=1)
y_predicted_labels = np.argmax(y_predicted, axis=1)

###evaluation of the model
accuracy = accuracy_score(y_test_labels, y_predicted_labels)
precision = precision_score(y_test_labels, y_predicted_labels, average='micro')
recall = recall_score(y_test_labels, y_predicted_labels, average='micro')
f1 = f1_score(y_test_labels, y_predicted_labels, average='micro')
roc_auc = roc_auc_score(y_test_labels, y_predicted_labels,average='micro')
#conf_matrix = confusion_matrix(y_test, y_predicted)


# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_labels, y_predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




