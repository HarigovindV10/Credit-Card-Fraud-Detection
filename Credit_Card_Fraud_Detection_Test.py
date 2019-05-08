import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import numpy as np


xgb = load('gbt.joblib')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')


predicted_array = xgb.predict(X_test)
accuracy = accuracy_score(y_test, predicted_array)
print("\nAccuracy : %.2f%%\n" % (accuracy * 100.0))


cm = confusion_matrix(y_test, predicted_array)
print(cm)


fraud_accuracy = cm[1][1]/(cm[1][1]+cm[0][1])
print("\nFraud Accuracy : %.2f%%\n" % (fraud_accuracy*100))


plt.figure(figsize = (9,5))
sn.heatmap(cm, annot=True,linewidth=0.5,fmt='g')
plt.show()
