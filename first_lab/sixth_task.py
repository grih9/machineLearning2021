from sklearn import tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

X_train = []
y_train = []
with open("1/bank_scoring_train.csv") as file:
    lines = file.readlines()
    names = lines[0].strip('\n').split("\t")[1:]
    lines = lines[1:]
    for line in lines:
        arr = line.strip('\n').split('\t')
        X_train.append(list(map(float, arr[1:])))
        y_train.append(int(arr[0]))

X_test = []
y_test = []
with open("1/bank_scoring_test.csv") as file:
    lines = file.readlines()
    names = lines[0].strip('\n').split("\t")[1:]
    lines = lines[1:]
    for line in lines:
        arr = line.strip('\n').split('\t')
        X_test.append(list(map(float, arr[1:])))
        y_test.append(int(arr[0]))

clf = tree.DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("testing accuracy:", end=" ")
print(metrics.accuracy_score(pred, y_test))
print(metrics.confusion_matrix(y_test, pred, labels=[0, 1]))

clf = GaussianNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("testing accuracy:", end=" ")
print(metrics.accuracy_score(pred, y_test))
print(metrics.confusion_matrix(y_test, pred, labels=[0, 1]))

clf = SVC(kernel="linear")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("testing accuracy:", end=" ")
print(metrics.accuracy_score(pred, y_test))
print(metrics.confusion_matrix(y_test, pred, labels=[0, 1]))