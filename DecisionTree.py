import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

balance_data = pd.read_csv('BIG.csv', sep= ',', header= None)


print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)

X = balance_data.values[:, 0:2]
Y = balance_data.values[:,-1]
#print(Y)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.1808, random_state = 100)


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
	
Z = clf_entropy.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axarr.contourf(xx, yy, Z, alpha=0.4)
axarr.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
plt.title('n=8192')


plt.savefig('8192.jpg', format = 'jpg', dpi = 1200)

plt.show()



y_pred_en = clf_entropy.predict(X_test)
print(y_pred_en)

	
print ("Accuracy is ", accuracy_score(y_test,y_pred_en))