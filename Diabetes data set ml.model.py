#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[4]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('C:/Users/Admin/Downloads.csv/diabetes (2).csv') 


# In[5]:


# printing the first 5 rows of the dataset
diabetes_dataset.head()


# In[6]:


# number of rows and Columns in this dataset
diabetes_dataset.shape


# In[7]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[8]:


diabetes_dataset['Outcome'].value_counts()


# In[9]:


diabetes_dataset.groupby('Outcome').mean()


# In[10]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
     


# In[11]:


print(Y)


# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
     

print(X.shape, X_train.shape, X_test.shape)


# In[13]:


classifier = svm.SVC(kernel='linear')
     

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[14]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
     

print('Accuracy score of the training data : ', training_data_accuracy)
     


# In[15]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
     


# In[16]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[17]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[18]:


import pickle
     

filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
     

# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))
     

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[19]:


for column in X.columns:
  print(column)
     


# In[ ]:




