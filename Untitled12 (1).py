#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[56]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[57]:


data = pd.read_csv("./articulos_ml.csv")
data.shape


# In[58]:


data.head()


# In[59]:


data.describe()


# In[60]:


data.drop(['Title','url', 'Elapsed days'],1).hist()
plt.show()


# In[61]:


filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]
 
colores=['orange','blue']
tamanios=[30,60]
 
f1 = filtered_data['Word count'].values
f2 = filtered_data['# Shares'].values
 

asignar=[]
for index, row in filtered_data.iterrows():
    if(row['Word count']>1808):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[62]:


dataX =filtered_data[["Word count"]]
X_train = np.array(dataX)
y_train = filtered_data['# Shares'].values
 

regr = linear_model.LinearRegression()
 

regr.fit(X_train, y_train)
 

y_pred = regr.predict(X_train)
 

print('Coefficients: \n', regr.coef_)

print('Independent term: \n', regr.intercept_)

print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))

print('Variance score: %.2f' % r2_score(y_train, y_pred))


# In[63]:


y_Dosmil = regr.predict([[2000]])
print(int(y_Dosmil))


# In[64]:


x = np.arange(-11,11)
y = 5.69765366*x + 11200.303223074163
plt.plot(x,y)


# In[ ]:




