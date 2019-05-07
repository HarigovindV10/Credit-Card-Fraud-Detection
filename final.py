
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[13]:


df = pd.read_csv("creditcard.csv")


# In[14]:


X = df.iloc[:,0:len(df.columns)-2].values
y = df.iloc[:,len(df.columns)-1].values


# In[15]:


from sklearn.decomposition import PCA
pca = PCA(n_components=15)
pca.fit(X)


# In[16]:


new_array = pca.fit_transform(X)


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[18]:


from sklearn.ensemble import GradientBoostingClassifier
xgb = GradientBoostingClassifier(loss = 'exponential',n_estimators=200)
xgb.fit(X_train,y_train)
predicted_array = xgb.predict(X_test)


# In[19]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_array)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[20]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_array)


# In[21]:


import seaborn as sn
sn.heatmap(cm, annot=True)


# In[22]:


print(cm)

