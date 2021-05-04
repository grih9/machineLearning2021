import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X = []
y = []

y_translator = {"van": 0,
                "saab": 1,
                "opel": 2,
                "bus": 3}

with open("4/vehicle.csv") as f:
    lines = f.readlines()
    headers = lines[0].strip('\n').replace('"', '').split(",")
    lines = lines[1:]
    for line in lines:
        arr = line.strip('\n').replace('"', '').split(",")
        X.append(list(map(int, arr[:-1])))
        y.append(y_translator[arr[-1]])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

kmax = int(len(y) ** 0.5) + 1

naive_beyes = GaussianNB()
svc = SVC()
decision_tree = DecisionTreeClassifier()

classifiers = [naive_beyes, svc, decision_tree]

for classifier in classifiers:
    yp = []
    xp = []
    for n in range(1, 50):
        ada_boost = AdaBoostClassifier(algorithm="SAMME", base_estimator=classifier, random_state=1, n_estimators=n)
        ada_boost.fit(X_train, y_train)
        pred = ada_boost.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        print(classifier, n, accuracy)
        yp.append(accuracy)
        xp.append(n)

    plt.plot(xp, yp, label=classifier)
    plt.xlabel("Classifiers number")
    plt.ylim(-0.2, 1)
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
plt.show()
