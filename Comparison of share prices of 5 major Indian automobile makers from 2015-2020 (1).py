#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


assets = ['TTM','M&M.NS','HEROMOTOCO.NS','MARUTI.NS','BAJAJ-AUTO.NS']
pf_data = pd.DataFrame()
for a in assets:
    pf_data[a] = wb.DataReader(a, data_source='yahoo',start = '2016-01-01')['Adj Close']


# In[5]:


pf_data.tail()


# In[6]:


pf_data.head()


# In[7]:


(pf_data/pf_data.iloc[1]* 100).plot(figsize=(15,10))


# In[8]:


log_returns = np.log(pf_data/pf_data.shift(1))


# In[9]:


log_returns.mean() * 250


# In[10]:


log_returns.cov() * 250


# In[11]:


log_returns.corr()


# In[12]:


n = len(assets)


# In[13]:


n


# In[14]:


arr = np.random.random(5)
arr


# In[15]:


arr[0]+arr[1]+arr[2]+arr[3]+arr[4]


# In[16]:


weights = np.random.random(n)
weights/= np.sum(weights)
weights 


# In[17]:


weights[0] + weights[1]+weights[2] + weights[3]+weights[4] 


# Expected Portfolio return

# In[18]:


np.sum(weights * log_returns.mean())*250


# Expected Portfolio variance

# In[19]:


np.dot(weights.T,np.dot(log_returns.cov()*250,weights))


# Expected Portfolio volatility

# In[20]:


np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*250,weights)))


# In[21]:


pfolio_returns = []
pfolio_volatilities = []
for x in range (1000):
    weights = np.random.random(n)
    weights/=np.sum(weights)
    pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
    pfolio_volatilities.append(np.dot(weights.T,np.dot(log_returns.cov()*250,weights)))
    
pfolio_returns,pfolio_volatilities    


# In[22]:


pfolio_returns = []
pfolio_volatilities = []
for x in range (1000):
    weights = np.random.random(n)
    weights/=np.sum(weights)
    pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
    pfolio_volatilities.append(np.dot(weights.T,np.dot(log_returns.cov()*250,weights)))
    
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)
pfolio_returns,pfolio_volatilities


# In[23]:


portfolios = pd.DataFrame({'Return':pfolio_returns,'Volatility':pfolio_volatilities})


# In[24]:


portfolios.head()


# In[25]:


portfolios.tail()


# In[36]:


portfolios.plot(x='Volatility',y='Return',kind='scatter',figsize=(10,6))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')


# In[40]:


x = pfolio_volatilities.min()
y = pfolio_returns.max()
print("Best case scenario")
print(x)
print(y)
print("Worst Possible scenario")
p = pfolio_volatilities.max()
q = pfolio_returns.min()
print(p)
print(q)


# Best Portfolio possibility : Return = 7.100 %
#                              Volatility in investment = 5.3904 %
#                              
# Worst Portfolio possibility : Return = -18.09 %
#                               Volatility in investment = 11.4639%

# In[ ]:




