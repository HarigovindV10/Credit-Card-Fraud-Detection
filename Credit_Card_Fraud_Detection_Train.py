import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from joblib import dump, load


df = pd.read_csv("creditcard.csv")
X = df.iloc[:,0:len(df.columns)-2].values
y = df.iloc[:,len(df.columns)-1].values


X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda x: x[0])
eig_pairs.reverse()
for i in eig_pairs:
    print(i[0])


matrix_w = np.hstack((eig_pairs[0][1].reshape(29,1),eig_pairs[1][1].reshape(29,1),eig_pairs[2][1].reshape(29,1),eig_pairs[3][1].reshape(29,1),eig_pairs[4][1].reshape(29,1),eig_pairs[5][1].reshape(29,1)))

matrix_final = X_std.dot(matrix_w)


def mahalanobis_distance(a,s):
    at = a.T
    b = np.dot(at,s)
    c = np.dot(b,a)
    return math.sqrt(c)


u = np.mean(matrix_final, axis=0)
s = (matrix_final - u).T.dot((matrix_final - u)) / (len(matrix_final)-1)
si = np.linalg.inv(s)
new_matrix =[]
for i in matrix_final:
    a = i-u
    new_matrix.append(mahalanobis_distance(a,si))


new_matrix_sorted = sorted(range(len(new_matrix)), key=lambda k: new_matrix[k])
per = int(len(new_matrix)*0.05)
matrix_reduced = []
for i in new_matrix_sorted[0:len(new_matrix_sorted)-per]:
    matrix_reduced.append(matrix_final[i])


"""for i in matrix_reduced[0:30000]:
    plt.scatter(i[0],i[1],color='b')
for j in new_matrix_sorted[len(new_matrix_sorted)-per:len(new_matrix_sorted)-per+3000]:
    plt.scatter(matrix_final[j][0],matrix_final[j][1],color='r')"""


X_train, X_test, y_train, y_test = train_test_split(X, y)



xgb = GradientBoostingClassifier(loss = 'exponential',n_estimators=200)
xgb.fit(X_train,y_train)


dump(xgb, 'gbt.joblib')
np.save('X_test',X_test)
np.save('y_test', y_test)
