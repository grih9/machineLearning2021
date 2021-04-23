import sklearn.tree as tree
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split


def task_a():
    X = []
    y = []
    with open("1/glass.csv") as file:
        lines = file.readlines()
        names = lines[0].strip('\n').replace('"', '').split(",")[1:-1]
        lines = lines[1:]
        for line in lines:
            arr = line.strip('\n').replace('"', '').split(",")
            X.append(list(map(float, arr[1:-1])))
            y.append(int(arr[-1]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    clf = tree.DecisionTreeClassifier(random_state=1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    for crit in ("gini", "entropy"):
        print(f"criterion={crit}")
        clf = tree.DecisionTreeClassifier(random_state=1, criterion=crit)
        clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print("testing accuracy:", end=" ")
        print(metrics.accuracy_score(pred, y_test))

    for splitter in ("best", "random"):
        print(f"splitter={splitter}")
        clf = tree.DecisionTreeClassifier(random_state=1, splitter=splitter)
        clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print("testing accuracy:", end=" ")
        print(metrics.accuracy_score(pred, y_test))

    acc1 = []
    for depth in range(1, 15):
        # print(f"depth={depth}, random_state=1")
        clf = tree.DecisionTreeClassifier(random_state=1, max_depth=depth)
        clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        # print("testing accuracy:", end=" ")
        # print(metrics.accuracy_score(pred, y_test))
        acc1.append(metrics.accuracy_score(pred, y_test))
    x = [n for n in range(1, 15)]
    # plt.title("random state=1")
    # plt.plot(x, acc)
    # plt.show()

    acc2 = []
    for depth in range(1, 15):
        # print(f"depth={depth}")
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        # print("testing accuracy:", end=" ")
        # print(metrics.accuracy_score(pred, y_test))
        acc2.append(metrics.accuracy_score(pred, y_test))

    plt.plot(x, acc1, label="random_state=1")
    plt.plot(x, acc2, label="no random_state")
    plt.legend()
    plt.show()

def task_b():
    X = []
    y = []
    y_translator = {
        "y": 0,
        "n": 1
    }
    with open("1/spam7.csv") as file:
        lines = file.readlines()
        names = lines[0].strip('\n').replace('"', '').split(",")[:-1]
        lines = lines[1:]
        for line in lines:
            arr = line.strip('\n').replace('"', '').split(",")
            X.append(list(map(float, arr[1:-1])))
            y.append(y_translator[arr[-1]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    clf = tree.DecisionTreeClassifier(random_state=1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    print("no random state")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(max_depth=15)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(max_depth=8)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(max_depth=7)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(max_depth=6)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(random_state=1, criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(random_state=1, max_depth=5, criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(max_depth=5, criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(max_features="sqrt", max_depth=5, criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(min_samples_split=1000, max_depth=5, criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(min_samples_split=200, max_depth=5, criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(min_samples_split=10, max_depth=5, criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

    clf = tree.DecisionTreeClassifier(min_samples_split=50, max_depth=5, criterion="entropy")
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    plt.figure(figsize=(60, 30))
    print(clf.get_depth())
    print(clf.get_params())
    tree.plot_tree(clf, feature_names=names,
                   filled=True, fontsize=5, rounded=True)
    plt.show()

#task_a()
task_b()