from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

x_translator = {"x": 0,
                "o": 1,
                "b": 2}

y_translator = {"positive": 0,
                "negative": 1}
X = list()
y = list()

with open("1/tic_tac_toe.txt", "r") as f:
    for line in f:
        arr = line.strip('\n').split(",")
        X.append([x_translator[elem] for elem in arr[:9]])
        y.append(y_translator[arr[9]])

xp = []
y1p = []
y2p = []

for train_s in range(5, 96, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_s / 100)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pr = model.predict(X_test)
    y2 = model.predict(X_train)
    print(len(y_train), len(y2))
    print(sum(y2 != y_train))
    print(1 - metrics.accuracy_score(y_train, y2))
    y1p.append(metrics.accuracy_score(y_train, y2))
    # print(len(y_test), len(y_pr))
    # print(sum(y_pr != y_test))
    # print(1 - metrics.accuracy_score(y_test, y_pr))
    y2p.append(metrics.accuracy_score(y_test, y_pr))
    xp.append(train_s / 100)

plt.plot(xp, y1p, label="train accuracy")
plt.plot(xp, y2p, label="test accuracy")
plt.xlabel("train size")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.show()