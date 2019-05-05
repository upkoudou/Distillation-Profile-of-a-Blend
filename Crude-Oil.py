
# coding: utf-8

# In[1]:


# impor packages  
from numpy import *
from matplotlib.pyplot import *
from pandas import *
from pandas import DataFrame

from matplotlib import style
style.use("ggplot")

import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#generating data in np arrays for use and manipulation
temp = np.array([45, 95, 101, 140, 179, 210, 225, 260, 310, 330, 360, 381])
Vpercent = np.array([ 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])


# In[3]:


#adding data into np array
df = np.array(list(zip(temp, Vpercent))).reshape(len(temp), 2)
df


# In[4]:


x= temp
y= Vpercent


# In[6]:


#show data
scatter(temp,Vpercent)
plt.xlabel('Temperature C')
plt.ylabel('Volume evaporated %')
plt.title('plot of the data')
show()


# In[8]:


#simple linear reg model, see fit
b, m = polyfit(x, y, 1)
fit = m*x + b
plt.plot(temp,Vpercent, 'bo')
plt.plot(x, fit,'r-')
plt.xlabel('Temperature C')
plt.ylabel('Volume evaporated %')
plt.title('linear regression fit')
plt.show()


# In[9]:


#Y of my fit, y hat.
print(fit)


# In[10]:


#results & Coefs trick
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

constant = Vpercent
X2 = sm.add_constant(constant)
est = sm.OLS(temp, X2)
est2 = est.fit()
print(est2.summary())


# In[11]:


#display my predicted values against my actual values 

print(fit)
print(Vpercent)


# In[ ]:


/*(model, Y, =, mx, +b, +, E)
where:
    -Y is the response variable, the volume of temperature
    -mx is the slope,
    -b is the intercep, value of the volume when the temprature is null
    -E are the residuals, the error term.

 There is a positivie relationship between the temperatue and the volume evaporated. As the temperature increases the
volume evaporated inscreases.

The conditions concerning the residuals are assumed to be validated. 
 R-squared: 0.990 the model is good enough for this use case*/

