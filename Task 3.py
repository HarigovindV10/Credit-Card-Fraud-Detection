
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[4]:


df = pd.read_csv("creditcard.csv")


# In[5]:


X = df.iloc[:,0:len(df.columns)-2].values
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
                      eig_pairs[1][1].reshape(29,1),))
matrix_final = X_std.dot(matrix_w)


# In[6]:


def mahalanobis_distance(a,s):
    at = a.T
    b = np.dot(at,s)
    c = np.dot(b,a)
    return math.sqrt(c)


# In[7]:


u = np.mean(matrix_final, axis=0)
s = (matrix_final - u).T.dot((matrix_final - u)) / (len(matrix_final)-1)
si = np.linalg.inv(s)
new_matrix =[]
for i in matrix_final:
    a = i-u
    new_matrix.append(mahalanobis_distance(a,si))


# In[8]:


new_matrix_sorted = sorted(range(len(new_matrix)), key=lambda k: new_matrix[k])


# In[9]:


per = int(len(new_matrix)*0.05)


# In[10]:


matrix_reduced = []
for i in new_matrix_sorted[0:len(new_matrix_sorted)-per]:
    matrix_reduced.append(matrix_final[i])


# In[11]:


print("Outliers Mahalanobis")
count_x = 0
count_y = 0
for i in new_matrix_sorted[len(new_matrix_sorted)-per:len(new_matrix_sorted)]:
    if y[i] == 1:
        count_y += 1
    else:
        count_x += 1
print("No of fraud :{0}\nNo of non-fraud: {1}\nPercentage of fraud :{2}%\n".format(count_y,count_x,(count_y/(count_y+count_x))*100))


# In[12]:


count_x = 0
count_y = 0
for i in new_matrix_sorted[0:len(new_matrix_sorted)-per]:
    if y[i] == 1:
        count_y += 1
    else:
        count_x += 1
print("No of fraud :{0}\nNo of non-fraud: {1}\nPercentage of fraud :{2}%\n".format(count_y,count_x,(count_y/(count_y+count_x))*100))


# In[13]:


a = 306/(306+186)
a = a*100
print("Percentage of fraud in outliers :{0}% ".format(a))


# In[20]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(matrix_final)
distances = kmeans.transform(matrix_final)
sorted_idx = np.argsort(distances.ravel())[::-1][:per]

new_X = np.delete(matrix_final, sorted_idx, axis=0)


# In[21]:


sorted_idx


# In[22]:


fo = 0
f = 0
for i in range(len(matrix_final)):
    if i in sorted_idx:
        if y[i] == 1:
            fo += 1
    elif y[i] == 1:
        f += 1
a = fo/(fo+f)
a = a*100
print("Percentage of fraud in outliers :{0}% ".format(a))
    

