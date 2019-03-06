
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random 


# In[2]:


df = pd.read_csv("creditcard.csv")


# In[3]:


df.head()


# In[4]:


X = df.iloc[:,0:len(df.columns)-2].values
amt =  df.iloc[:,len(df.columns)-2:len(df.columns)-1].values
y = df.iloc[:,len(df.columns)-1].values


# In[5]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# In[6]:


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)


# In[7]:


eig_vals, eig_vecs = np.linalg.eig(cov_mat)


# In[8]:


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda x: x[0])
eig_pairs.reverse()
for i in eig_pairs:
    print(i[0])


# In[9]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(29,1), 
                     eig_pairs[1][1].reshape(29,1), 
                     eig_pairs[2][1].reshape(29,1), 
                     eig_pairs[3][1].reshape(29,1), 
                     eig_pairs[4][1].reshape(29,1), 
                     eig_pairs[5][1].reshape(29,1)))

#print(matrix_w)


# In[10]:


matrix_new = X_std.dot(matrix_w)


# In[11]:


matrix_final = []
for i,j in zip(matrix_new,amt):
   i = np.append(i, j)
   matrix_final.append(i)


# In[35]:


centers = []
for i in range(6):
    a = random.randint(0,len(matrix_final))
    centers.append(matrix_final[a])


# In[36]:


centers


# In[37]:


clusters = []
for i in centers:
    clusters.append([])


# In[38]:


def euclidean_distance(x,y):
    count = 0
    #print(x,y)
    for i,j in zip(x,y):
        
        count += (i - j) * (i - j)
    return math.sqrt(count)


# In[39]:


def min_distance(x):
    return x.index(min(x))


# In[50]:


min_list = []
while(True):
    for i in range(len(matrix_final)):
        for j in centers:
            min_list.append(euclidean_distance(matrix_final[i],j))
        l = min_distance(min_list)
        clusters[l].append(i)
        min_list = [] 
    temp = []
    for i in clusters:
        d = 0
        count  = 0
        for j in i:
            d += matrix_final[j]
            count += 1
        temp.append(d/count)
    flag = 1
    for i,j in zip(centers,temp):
        if (i == j).all:
            flag = 0
    if flag == 0:
        break
    centers = np.copy(temp)
    
        


# In[51]:


for i in range(len(clusters)):
    print("Cluster {}".format(i+1))
    for j in range(10):
        print(matrix_final[clusters[i][j]])


# In[52]:


a = []
plt.figure(figsize=(20,30))
plt.yticks(np.arange(0, 25000, 5000))
plt.xticks(np.arange(0, 284807, 20000))
for i in range(0,len(clusters)):
    for j in clusters[i]:
        a.append(matrix_final[j][6])
    plt.scatter(clusters[i],a)
    a=[]

