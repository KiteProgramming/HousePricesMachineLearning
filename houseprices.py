#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Necessary modules.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#passing our data into the data variable

data = pd.read_csv ('kc_house_data.csv')


# In[3]:


#Gathering info about our data.

data.info()


# In[4]:


#printing the data in graphs in order to choose between which ones to use as features for our prediction 

#using history module for plotting the data

fig = data
h = fig.hist(bins=25,figsize=(20,20),xlabelsize='15',ylabelsize='15',xrot=-15)
sns.despine(left=True, bottom=True)
[x.title.set_size(12) for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];


# In[5]:


# Plotting 6 possible features that will be used for the price prediction of prices using boxplot. 


f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=data['bedrooms'],y=data['price'], ax=axes[0])
sns.boxplot(x=data['floors'],y=data['price'], ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='Bedrooms', ylabel='Price')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='floors', ylabel='Price')

f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=data['bathrooms'],y=data['price'], ax=axes[0])
sns.boxplot(x=data['view'],y=data['price'], ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='bathrooms', ylabel='Price')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='view', ylabel='Price')

f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=data['condition'],y=data['price'], ax=axes[0])
sns.boxplot(x=data['sqft_living'],y=data['price'], ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='condition', ylabel='Price')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='sqft_living', ylabel='Price')


# In[6]:


"""Creating a correlation matrix between the data in order to properly choose which of the features that we have chosen above 
are not a good match to be used as features because of the fact that if two features are high correlated they may cause 
overfitting.

In the first line of code features = data would not work properly for .corr() function so I had to pass all the data separately 
for the features[] array.
"""

features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
            'zipcode','lat','long','sqft_living15','sqft_lot15']

mask = np.zeros_like(data[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(18, 13))
plt.title('Data Correlation Matrix',fontsize=35)

sns.heatmap(data[features].corr(),linewidths=0.3,vmax=0.8,square=True,cmap="Blues", 
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9})


# In[7]:


#Creating a copy of data so we can use it to change and split the data

data_copy = data.copy()


# In[8]:


#Gathering information to see necessary changes on our data. 


data_copy.info()

#We can see here that the bathrooms and floors feature have the float data type but we need to change it to integer.


# In[9]:


#Changing bathrooms and floors features to integer datatyupe for the purpose of memory saving and then below showing the results.

data_copy.bathrooms = data_copy.bathrooms.astype(int)
data_copy.floors = data_copy.floors.astype(int)


# In[10]:


data_copy.info()


# In[11]:


#Checking for our target and our features if there are any missing values.



data_copy.isnull().sum() # This way proved to be more sufficient


# In[12]:


#Implementing KNN Regression



data_copy.keys()


#PANDAS


X = data_copy[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']] 
y = data_copy[['price']]


# split dataset

X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.25 , random_state = 0)

#fit


model = KNeighborsRegressor(n_neighbors=8)



model.fit(X_train , y_train)

#prediction model
y_pred = model.predict(X_test)



r2_score(y_test,y_pred)



MSE = mean_squared_error(y_test,y_pred) 

y_train.shape
X_test.shape

# print(y_train.shape)
# print(X_test.shape)

print('This is r2_score:',round(r2_score(y_test,y_pred),2))

print('This is our Mean Squared Error:',round(MSE,2))


# In[13]:


#Scaling Data and comparing MSE before and after scale.

scaled_features = StandardScaler()
scaled_target = StandardScaler()

X_train_scale = scaled_features.fit_transform(X_train)
X_test_scale = scaled_features.transform(X_test)
y_train_scale = scaled_features.fit_transform(y_train)
y_test_scale = scaled_features.transform(y_test)


# In[14]:


#Implementing Linear Regression without scaled data

model2 = LinearRegression()
model2.fit(X_train,y_train)
score = model.score(X_test,y_test)
rmse = mean_squared_error(y_test,y_pred)
print('This is the score:',round(score , 2))
print('This is Root Mean Squared Error:', round(rmse , 2))


# In[15]:


#Implementing KNN Regression for the purpose of having two models for this project with the scaled data

model = KNeighborsRegressor(n_neighbors=8)
 

scores = cross_val_score(model,X_train_scale , y_train_scale,  cv = 5 , scoring = 'r2')
MSE = mean_squared_error(y_test_scale,y_pred)
print('cross validation score KNN:',round(np.mean(scores),2))

print('This is Mean Squared Error:',round(MSE , 2))

model = KNeighborsRegressor(n_neighbors=8)

model.fit(X_train_scale,y_train_scale)

y_pred = model.predict(X_test_scale)

print('Test Score for KNN non tune model:',round(r2_score(y_test_scale,y_pred), 2))


# In[16]:


#Implementing Linear Regression with scaled data

model2 = LinearRegression()
scores = cross_val_score(model2,X_train_scale , y_train_scale,  cv = 5 , scoring = 'r2')
rmse = mean_squared_error(y_test_scale,y_pred)
print('This is cross val score Linear:',round(np.mean(scores),2))
print('This is Root Mean Squared Error:', round(rmse , 2))

model2 = LinearRegression()

model2.fit(X_train_scale,y_train_scale)

y_pred = model2.predict(X_test_scale)

print('Test Score for Linear non tune model:',round(r2_score(y_test_scale,y_pred), 2))


# In[17]:


# Applying Hyper Parameter Tuning with GridSearchCV for Linear Regression

model = LinearRegression()
nrange = list(range(1,35))
fit_bool = [True, False]
normalize_bool = [True,False]
copy_X_bool = [True,False]
parametergrid = dict(copy_X = copy_X_bool , fit_intercept = fit_bool , n_jobs = nrange , normalize = normalize_bool)
gd = GridSearchCV(model,parametergrid,cv = 5,scoring = 'r2',return_train_score = False)
gd.fit(X_train_scale , y_train_scale)

print('Best Score:',round(gd.best_score_ , 2))
print(gd.best_params_)

y_pred = gd.best_estimator_.predict(X_test_scale)

print('Test Score for Linear tune model:',round(r2_score(y_test_scale,y_pred), 2))


# In[18]:


# Applying Hyper Parameter Tuning with GridSearchCV for KNN Regression

model = KNeighborsRegressor()
nrange = list(range(1,35))
wrange = ['uniform','distance']
parametergrid = dict(n_neighbors = nrange , weights = wrange)
gd = GridSearchCV(model,parametergrid,cv = 5,scoring = 'r2',return_train_score = False)
gd.fit(X_train_scale , y_train_scale)

print('Best Score',round(gd.best_score_ , 2))
print(gd.best_params_)

y_pred = gd.best_estimator_.predict(X_test_scale)

print('Test Score for KNN tune model:',round(r2_score(y_test_scale,y_pred), 2))






