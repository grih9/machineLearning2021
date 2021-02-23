import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split


X = []
y = []

test_example_X = [[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]]

with open("1/glass.csv") as f:
    lines = f.readlines()[1:]
    for line in lines:
        arr = line.strip('\n').replace('"', '').split(",")
        X.append(list(map(float, arr[1:-1])))
        y.append(int(arr[-1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
kmax = int(len(y) ** 0.5) + 1

xp = []
y1p = []
y2p = []
test_example_y = [0 for _ in range(7)]

for k in range(1, kmax + 1):
    nbrs = KNeighborsClassifier(n_neighbors=k)
    nbrs.fit(X_train, y_train)
    predicted = nbrs.predict(X_test)
    y2p.append(1 - metrics.accuracy_score(y_test, predicted))
    predicted = nbrs.predict(X_train)
    y1p.append(1 - metrics.accuracy_score(y_train, predicted))
    xp.append(k)
    test_example_y[nbrs.predict(test_example_X)[0] - 1] += 1

for metric_ in ("euclidean", "manhattan", "chebyshev", "minkowski"):
    nbrs = KNeighborsClassifier(n_neighbors=kmax, metric=metric_)
    print(metric_)
    nbrs.fit(X_train, y_train)
    predicted = nbrs.predict(X_test)
    print("test ", metrics.accuracy_score(y_test, predicted))
    predicted = nbrs.predict(X_train)
    print("train ", metrics.accuracy_score(y_train, predicted))
    test_example_y[nbrs.predict(test_example_X)[0] - 1] += 1

print(test_example_y)
plt.plot(xp, y1p, label="train error")
plt.plot(xp, y2p, label="test error")
plt.xlabel("k neighbours")
plt.ylabel("error rate")
plt.legend(loc="best")
plt.show()