#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,  mean_squared_error


# In[28]:


data = pd.read_csv('C:\\Users\\Jose\\Downloads\\RoadAccidents.csv')
data


# In[29]:


print("Shape of the dataset:", data.shape)


# In[30]:


data.describe()


# In[31]:


# Split the data into features (X) and the target variable (y)


# In[34]:


X = df[['Number_of_vehicles_involved', 'Number_of_casualties', 'Road Conditions', 'Weather Conditions']]
y = df['Accident Severity']


# In[35]:


# Split the data into a training set and a test set


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


# Create and train the linear regression model


# In[38]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[39]:


# Make predictions on the test set


# In[40]:


y_pred = model.predict(X_test)


# In[41]:


y_pred


# In[42]:


# Evaluate the model


# In[43]:


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[44]:


#Accuracy


# In[46]:


accuracy = mean_absolute_error(y_test,y_pred)
accuracy


# In[47]:


#Save the model fo future use


# In[48]:


import joblib


# In[49]:


joblib.dump(model,'Road_Accident_Severity.pkl')

