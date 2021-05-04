import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X = []
y = []

with open("4/glass.csv") as f:
    lines = f.readlines()
    headers = lines[0].strip('\n').replace('"', '').split(",")
    lines = lines[1:]
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

k_neighbours1 = KNeighborsClassifier()
k_neighbours2 = KNeighborsClassifier(n_neighbors=kmax)
naive_beyes = GaussianNB()
svc = SVC()
decision_tree = DecisionTreeClassifier()

classifiers = [k_neighbours1, k_neighbours2, naive_beyes, svc, decision_tree]

for classifier in classifiers:
    yp = []
    xp = []
    for n in range(1, 50):
        bagging = BaggingClassifier(base_estimator=classifier, random_state=1, n_estimators=n)
        bagging.fit(X_train, y_train)
        pred = bagging.predict(X_test)
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
