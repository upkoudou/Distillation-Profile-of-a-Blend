
# coding: utf-8

# In[47]:


# impor packages  
from numpy import *
from matplotlib.pyplot import *
from pandas import *
from pandas import DataFrame
import csv
from matplotlib import style
style.use("ggplot")
from scipy.interpolate import *
from sklearn import datasets
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[33]:


pwd()


# In[128]:


import pandas as pd
data = pd.read_csv('crude-file.csv')
data.head(10)


# In[166]:


data.describe()


# In[325]:


#plot the distribution of the data 
data.plot(kind='hist', alpha=.4, legend=True) # alpha for transparency
plt.xlabel('Temperature C')
plt.ylabel('Volume evaporated %')
plt.title('Distribution') 


# In[320]:


#show data
y_profile = data.Volume
x_profile = data.temperature

plt.scatter(x_profile, y_profile)
plt.xlabel('Temperature C')
plt.ylabel('Volume evaporated %')
plt.title('Volume evaporation by temperature') 
plt.show


# In[322]:


#simple linear reg model, see fit
b, m = polyfit(x_profile, y_profile, 1)
fit = m*x_profile + b
plt.plot(x_profile, y_profile, 'bo')
plt.plot(x_profile, fit,'r-')
plt.xlabel('Temperature C')
plt.ylabel('Volume evaporated %')
plt.title('Linear Regression fit') 


# In[277]:


#results & Coefs trick
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

constant = y_profile
y = sm.add_constant(constant)
est = sm.OLS(x_profile, y)
est2 = est.fit()
print(est2.summary())


# In[302]:


X = data.temperature.values.reshape(len(data.temperature.values), 1)
y = data.Volume.values.reshape(len(data.temperature.values), 1)


# In[317]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='blue')
    plt.plot(X, lin_reg.predict(X), color='red')
    plt.xlabel('Temperature C')
    plt.ylabel('Volume evaporated %')
    plt.title('Linear Regression fit') 
    plt.show()
    return
viz_linear()


# In[346]:


#Linear model coef of goodness/Determination

r_squared = coefficient_of_determination(X, y)
print(r_squared)


# In[340]:


# Even though the R**2 valu is more satisfying at a glance we can easily deduct that the simple linear model
#isn't the best fit, let's try with a polynomial model to the nTh degree until best fit. 

# Visualising the Polynomial Regression results 

poly = PolynomialFeatures(degree = 5) 
X_poly = poly.fit_transform(X)  

plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.xlabel('Temperature C')
plt.ylabel('Volume evaporated %')
plt.title('Polynomial Regression Fit') 
  
plt.show() 


# In[341]:


#Polynomial model coef of goodness/Determination at order 5
from sklearn.metrics import r2_score
y_true = y_profile 
y_pred = y


# In[342]:


r2_score(y_true, y_pred) ##1 ??


# In[ ]:


*the(Polynomial, model, fit, our, data, better,, the, R**2, value, has, also, increased,, the, next, step, would, have, been, to, look)
for the limitations of the model, by splitting my dataset two parts, testing and trainnig. 
Train the model and test the model on the data to appreciate its accuracy.

The conditions concerning the residuals are assumed to be validated. 
*/

