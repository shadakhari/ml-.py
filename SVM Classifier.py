#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[8]:


diabetes_dataset =pd.read_csv("C:/Users/Admin/Downloads.csv/diabetes.csv")
diabetes_dataset


# In[11]:


diabetes_dataset.head()


# In[14]:


diabetes_dataset.shape


# In[15]:


diabetes_dataset.describe()


# In[17]:


diabetes_dataset['Outcome'].value_counts()


# In[19]:


diabetes_dataset.groupby('Outcome').mean()


# In[21]:


x=diabetes_dataset.drop(axis=1,columns='Outcome')


# In[24]:


y=diabetes_dataset['Outcome']


# In[26]:


print(x)


# In[27]:


print(y)


# In[29]:


scaler=StandardScaler()


# In[32]:


scaler.fit(x)


# In[36]:


standardized_data = scaler.transform(x)


# In[37]:


print(standardized_data)


# In[38]:


x=standardized_data
y=diabetes_dataset['Outcome']


# In[39]:


print(x,y)


# In[43]:


X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, stratify=y, random_state=2)


# In[45]:


print(x.shape, X_train.shape, X_test.shape)


# In[46]:


classifier = svm.SVC(kernel='linear')


# In[47]:


classifier.fit(X_train, Y_train)


# In[48]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[49]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[50]:


input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




