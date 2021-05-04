import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = [], [], [], []

gender_translator = {"male": 0,
                     "female": 1}

embarked_translator = {"None": -1,
                       "C": 0,
                       "S": 1,
                       "Q": 2}

with open("4/titanic_train.csv") as f:
    lines = f.readlines()
    headers = lines[0].strip('\n').replace('"', '').split(",")
    lines = lines[1:]
    for line in lines:
        arr = line.strip('\n').replace('"', '').split(",")
        tmp = list()
        tmp.append(int(arr[2]))
        tmp.append(gender_translator[arr[5]])
        if arr[6] == "":
            continue
            tmp.append(35.0)
        else:
            tmp.append(float(arr[6]))
        tmp += list(map(int, arr[7:9]))
        tmp.append(float(arr[10]))
        if arr[12] == "":
            continue
            tmp.append(embarked_translator["None"])
        else:
            tmp.append(embarked_translator[arr[12]])
        X_train.append(tmp)
        y_train.append(arr[1])

print(len(X_train))
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.75)
#X_test, y_test = X_train, y_train

svc = SVC()
k_neighb = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier()
naive_buyes = GaussianNB()

classifiers41 = [('kn', k_neighb),
                 ('dt', decision_tree),
                 ('nb', naive_buyes),
                 ('svc', svc)]

classifiers31 = [('kn', k_neighb),
                 ('dt', decision_tree),
                 ('nb', naive_buyes)]

classifiers32 = [('kn', k_neighb),
                 ('dt', decision_tree),
                 ('svc', svc)]

classifiers33 = [('kn', k_neighb),
                 ('nb', naive_buyes),
                 ('svc', svc)]

classifiers34 = [('dt', decision_tree),
                 ('nb', naive_buyes),
                 ('svc', svc)]

classifiers21 = [('kn', k_neighb),
                 ('nb', naive_buyes)]
classifiers22 = [('dt', decision_tree),
                 ('nb', naive_buyes)]
classifiers23 = [('kn', k_neighb),
                 ('dt', decision_tree)]
classifiers24 = [('kn', k_neighb),
                 ('svc', svc)]
classifiers25 = [('dt', decision_tree),
                 ('svc', svc)]
classifiers26 = [('nb', naive_buyes),
                 ('svc', svc)]

print("4 values:")
stack1 = StackingClassifier(estimators=classifiers41)
stack1.fit(X_train, y_train)

pred = stack1.predict(X_test)
print(f"{[elem[0] for elem in classifiers41]} -", metrics.accuracy_score(y_test, pred))

print("3 values:")
stack2 = StackingClassifier(estimators=classifiers31)
stack2.fit(X_train, y_train)

pred = stack2.predict(X_test)
print(f"{[elem[0] for elem in classifiers31]} -", metrics.accuracy_score(y_test, pred))

stack3 = StackingClassifier(estimators=classifiers32)
stack3.fit(X_train, y_train)

pred = stack3.predict(X_test)
print(f"{[elem[0] for elem in classifiers32]} -", metrics.accuracy_score(y_test, pred))

stack4 = StackingClassifier(estimators=classifiers33)
stack4.fit(X_train, y_train)

pred = stack4.predict(X_test)
print(f"{[elem[0] for elem in classifiers33]} -", metrics.accuracy_score(y_test, pred))

stack5 = StackingClassifier(estimators=classifiers34)
stack5.fit(X_train, y_train)

pred = stack5.predict(X_test)
print(f"{[elem[0] for elem in classifiers34]} -", metrics.accuracy_score(y_test, pred))

print("2 values:")
stack1 = StackingClassifier(estimators=classifiers21)
stack1.fit(X_train, y_train)

pred = stack1.predict(X_test)
print(f"{[elem[0] for elem in classifiers21]} -", metrics.accuracy_score(y_test, pred))

stack2 = StackingClassifier(estimators=classifiers22)
stack2.fit(X_train, y_train)

pred = stack2.predict(X_test)
print(f"{[elem[0] for elem in classifiers22]} -", metrics.accuracy_score(y_test, pred))

stack3 = StackingClassifier(estimators=classifiers23)
stack3.fit(X_train, y_train)

pred = stack3.predict(X_test)
print(f"{[elem[0] for elem in classifiers23]} -", metrics.accuracy_score(y_test, pred))

stack4 = StackingClassifier(estimators=classifiers24)
stack4.fit(X_train, y_train)

pred = stack4.predict(X_test)
print(f"{[elem[0] for elem in classifiers24]} -", metrics.accuracy_score(y_test, pred))

stack5 = StackingClassifier(estimators=classifiers25)
stack5.fit(X_train, y_train)

pred = stack5.predict(X_test)
print(f"{[elem[0] for elem in classifiers25]} -", metrics.accuracy_score(y_test, pred))

stack6 = StackingClassifier(estimators=classifiers26)
stack6.fit(X_train, y_train)

pred = stack6.predict(X_test)
print(f"{[elem[0] for elem in classifiers26]} -", metrics.accuracy_score(y_test, pred))

print("1 value:")
k_neighb = KNeighborsClassifier()
k_neighb.fit(X_train, y_train)

pred = k_neighb.predict(X_test)
print("k_neigb -", metrics.accuracy_score(y_test, pred))

svc = SVC()
svc.fit(X_train, y_train)

pred = svc.predict(X_test)
print("svc -", metrics.accuracy_score(y_test, pred))


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

pred = decision_tree.predict(X_test)
print("dt -", metrics.accuracy_score(y_test, pred))

naive_buyes = GaussianNB()
naive_buyes.fit(X_train, y_train)

pred = naive_buyes.predict(X_test)
print("nb -", metrics.accuracy_score(y_test, pred))
