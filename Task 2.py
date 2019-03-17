
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# In[4]:


df = pd.read_csv("creditcard.csv")


# In[5]:


X = df.iloc[:,0:len(df.columns)-2].values
amt =  df.iloc[:,len(df.columns)-2:len(df.columns)-1].values
y = df.iloc[:,len(df.columns)-1].values
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda x: x[0])
eig_pairs.reverse()
matrix_w = np.hstack((eig_pairs[0][1].reshape(29,1), 
                      eig_pairs[1][1].reshape(29,1), 
                      eig_pairs[2][1].reshape(29,1), 
                      eig_pairs[3][1].reshape(29,1), 
                      eig_pairs[4][1].reshape(29,1), 
                      eig_pairs[5][1].reshape(29,1)))
matrix_new = X_std.dot(matrix_w)
matrix_final = []
for i,j in zip(matrix_new,amt):
    i = np.append(i, j)
    matrix_final.append(i)


# In[41]:


def mahalanobis_distance(a,s):
    at = a.T
    b = np.dot(at,s)
    c = np.dot(b,a)
    return math.sqrt(c)


# In[42]:


u = np.mean(matrix_final, axis=0)
s = (matrix_final - u).T.dot((matrix_final - u)) / (len(matrix_final)-1)
si = np.linalg.inv(s)
new_matrix =[]
for i in matrix_final:
    a = i-u
    new_matrix.append(mahalanobis_distance(a,si))


# In[46]:


new_matrix_sorted = sorted(range(len(new_matrix)), key=lambda k: new_matrix[k])


# In[50]:


per = int(len(new_matrix)*0.05)


# In[60]:


matrix_reduced = []
for i in new_matrix_sorted[0:len(new_matrix_sorted)-per]:
    matrix_reduced.append(matrix_final)


# In[62]:




