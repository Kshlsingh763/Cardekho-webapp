#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


import pickle


# In[2]:


# importing data

data=pd.read_csv('car data.csv')


# In[3]:


# dropping feature-Car Name as this has no role in predicting prices

data=data.drop(columns='Car_Name')


# In[4]:


# creating dummies for categorical features

data=pd.get_dummies(data,drop_first=True)


# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


X=data.drop('Selling_Price',axis=1)
y=data['Selling_Price']


# In[7]:


lr=LinearRegression()


# In[8]:


lr.fit(X,y)


# In[10]:


# Saving model to disk
pickle.dump(lr, open('model.pkl','wb'))


# In[11]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




