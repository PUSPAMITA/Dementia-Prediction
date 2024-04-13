#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.model_selection import train_test_split  ##for splitting training and test data
from sklearn.preprocessing import StandardScaler   ##feature scaling
##for assesing the accuracy of our model
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


# In[2]:


df = pd.read_csv(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv")
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df=df.drop('Prescription',axis=1)
df=df.drop('Dosage in mg', axis=1)


# In[7]:


### Replace non-numerical values with integers in columns
### Replace non-numerical values with integers in columns
### Define a function to replace non-numerical values with integers

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



# In[8]:


df.head()


# In[9]:


##Feature Scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Dementia', axis=1)
y = df['Dementia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Convert Pandas Series to NumPy array
y_train_array = np.array(y_train)

# Reshape the array
y_train_reshaped = y_train_array.reshape((800, 1))

# Convert Pandas Series to NumPy array
y_test_array = np.array(y_test)

# Reshape the array
y_test_reshaped = y_test_array.reshape((200, 1))


# In[10]:


df.info()


# In[11]:


import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    
    Parameters:
    z (numpy.ndarray): Input array
    
    Returns:
    numpy.ndarray: Sigmoid activated output
    """
    return 1/(1+np.exp(-z))

def train_logistic_regression(learning_rate, epochs, initial_weight, initial_bias, X_train_scaled, y_train_reshaped):
    """
    Train logistic regression using gradient descent.
    
    Parameters:
    learning_rate (float): Learning rate for gradient descent
    epochs (int): Number of training epochs
    initial_weight (numpy.ndarray): Initial weight vector
    initial_bias (float): Initial bias value
    X_train_scaled (numpy.ndarray): Scaled feature matrix of shape (num_samples, num_features)
    y_train_reshaped (numpy.ndarray): Reshaped target vector of shape (num_samples, 1)
    
    Returns:
    numpy.ndarray: Learned weight vector
    float: Learned bias value
    """
    num_samples, num_features = X_train_scaled.shape
    
    weight = initial_weight
    bias = initial_bias
    
    for epoch in range(epochs):
        # Forward pass
        z = np.dot(X_train_scaled, weight) + bias
        predicted = sigmoid(z)
        
        # Clip predicted values to avoid log(0) and log(1) issues
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        
        # Loss computation
        loss = -(1/num_samples) * np.sum(y_train_reshaped*np.log(predicted) + (1-y_train_reshaped)*np.log(1-predicted))
        
        # Gradient computation
        d_wt = (1/num_samples) * np.dot(X_train_scaled.T, (predicted - y_train_reshaped))
        d_bias = (1/num_samples) * np.sum(predicted - y_train_reshaped)
        
        # Update weights and bias
        weight -= learning_rate * d_wt
        bias -= learning_rate * d_bias
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss {loss}')
    
    return weight, bias

# Example usage:
# Set your hyperparameters
learning_rate = 0.01
epochs = 1000

# Initialize weights and bias
initial_weight = np.random.rand(X_train_scaled.shape[1], 1)
initial_bias = np.random.rand()

# Train logistic regression model
learned_weight, learned_bias = train_logistic_regression(learning_rate, epochs, initial_weight, initial_bias, X_train_scaled, y_train_reshaped)


# In[16]:


def predict(X_test_scaled, weight, bias):
    """
    Predict probabilities using the trained logistic regression model.
    
    Parameters:
    X_test_scaled (numpy.ndarray): Scaled feature matrix of shape (num_samples, num_features)
    weight (numpy.ndarray): Learned weight vector
    bias (float): Learned bias value
    
    Returns:
    numpy.ndarray: Predicted probabilities of shape (num_samples, 1)
    """
    z = np.dot(X_test_scaled, weight) + bias
    return sigmoid(z)

# Assuming you have your test set X_test_scaled and y_test_reshaped ready

# Predict probabilities for the test set
y_pred_prob = predict(X_test_scaled, learned_weight, learned_bias)
#print(y_pred_prob)
# Convert probabilities to binary predictions (0 or 1)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Assuming y_test_reshaped is a binary vector
#accuracy = np.mean(y_pred == y_test_reshaped)
#print("Test accuracy:", accuracy)


# In[17]:


import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

###evaluation of the model
accuracy = accuracy_score(y_test_reshaped, y_pred)
precision = precision_score(y_test_reshaped, y_pred)
recall = recall_score(y_test_reshaped, y_pred)
f1 = f1_score(y_test_reshaped, y_pred)
roc_auc = roc_auc_score(y_test_reshaped, y_pred)
conf_matrix = confusion_matrix(y_test_reshaped, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.metrics import roc_curve, roc_auc_score
# Assuming you have true labels and predicted class probabilities
fpr, tpr, thresholds = roc_curve(y_test_reshaped, y_pred)


# In[18]:


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
plt.figure(figsize=(6,6 ))
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

