def task_a():
    print("task 4a")
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn import metrics
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    translator = {"red": 1,
                  "green": 2}

    with open("1/svmdata_a.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            arr = line.strip('\n').replace('"', '').split("\t")
            X_train.append(list(map(float, arr[1:-1])))
            y_train.append(translator[arr[-1]])

    with open("1/svmdata_a_test.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            arr = line.strip('\n').replace('"', '').split("\t")
            X_test.append(list(map(float, arr[1:-1])))
            y_test.append(translator[arr[-1]])

    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("testing accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_test))
    print(metrics.confusion_matrix(y_test, pred))
    pred = clf.predict(X_train)
    print("training accuracy:", end=" ")
    print(metrics.accuracy_score(pred, y_train))
    print(metrics.confusion_matrix(y_train, pred))
    print(clf.support_vectors_)
    print(clf.n_support_)
    X0 = [elem[0] for elem in X_train]
    X1 = [elem[1] for elem in X_train]
    plt.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=70)
    X0t = [elem[0] for elem in X_test]
    X1t = [elem[1] for elem in X_test]
    plt.scatter(X0t, X1t, c=y_test, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("linear")

    xx = np.array(X_train)
    yy = np.array(y_train)
    xx = xx.astype(np.float64)
    yy = yy.astype(np.int32)
    X0, X1 = xx[:, 0], xx[:, 1]
    xx, yy = np.meshgrid(np.arange(X0.min() - 0.5, X0.max() + 0.5, .1),
                         np.arange(X1.min() - 0.5, X1.max() + 0.5, .1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def task_b():
    print("task 4b")
    from sklearn.svm import SVC
    from sklearn import metrics
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    translator = {"red": 1,
                  "green": 2}

    with open("1/svmdata_b.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            arr = line.strip('\n').replace('"', '').split("\t")
            X_train.append(list(map(float, arr[1:-1])))
            y_train.append(translator[arr[-1]])

    with open("1/svmdata_b_test.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            arr = line.strip('\n').replace('"', '').split("\t")
            X_test.append(list(map(float, arr[1:-1])))
            y_test.append(translator[arr[-1]])

    min = 1000
    max = 0
    for c in range(1, 1000):
        clf = SVC(kernel="linear", C=c)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        testing = metrics.accuracy_score(pred, y_test)
        if testing != 1 and max == 0:
            max = c - 1
        pred = clf.predict(X_train)
        training = metrics.accuracy_score(pred, y_train)
        if (training == 1.0):
            min = c
            break

    for type in (max, min):
        clf = SVC(kernel="linear", C=type)
        print(type)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print("testing accuracy:", end=" ")
        print(metrics.accuracy_score(pred, y_test))
        print(metrics.confusion_matrix(y_test, pred))
        pred = clf.predict(X_train)
        print("training accuracy:", end=" ")
        print(metrics.accuracy_score(pred, y_train))
        print(metrics.confusion_matrix(y_train, pred))


def task_cde(task, gamma="scale", c=1):
    print("\ntask 4" + task + str(gamma) + str(c))
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn import metrics
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    translator = {"red": 1,
                  "green": 2}

    with open("1/svmdata_" + task + ".txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            arr = line.strip('\n').replace('"', '').split("\t")
            X_train.append(list(map(float, arr[1:-1])))
            y_train.append(translator[arr[-1]])

    with open("1/svmdata_" + task + "_test.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            arr = line.strip('\n').replace('"', '').split("\t")
            X_test.append(list(map(float, arr[1:-1])))
            y_test.append(translator[arr[-1]])

    if task == "c":
        models = (SVC(kernel="linear"),
                  SVC(kernel="poly", gamma=gamma, degree=1, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=2, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=3, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=4, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=5, C=c),
                  SVC(kernel="rbf", gamma=gamma, C=c),
                  SVC(kernel="sigmoid", gamma=gamma, C=c))
        fig, sub = plt.subplots(4, 2)
    elif task == "d":
        models = (SVC(kernel="poly", gamma=gamma, degree=1, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=2, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=3, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=4, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=5, C=c),
                  SVC(kernel="rbf", gamma=gamma, C=c),
                  SVC(kernel="sigmoid", gamma=gamma, C=c))
        fig, sub = plt.subplots(4, 2)
    elif task == "e":
        models = (SVC(kernel="poly", gamma=gamma, degree=1, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=2, C=c),
                  SVC(kernel="poly", gamma=gamma, degree=4, C=c),
                  SVC(kernel="rbf", gamma=gamma, C=c),
                  SVC(kernel="sigmoid", gamma=gamma, C=c))
        fig, sub = plt.subplots(3, 2)

    for model, ax in zip(models, sub.flatten()):
        print(model.kernel, model.degree)
        clf = model
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print("testing accuracy:", end=" ")
        print(metrics.accuracy_score(pred, y_test))
        pred = clf.predict(X_train)
        print("training accuracy:", end=" ")
        print(metrics.accuracy_score(pred, y_train))
        X0 = [elem[0] for elem in X_train]
        X1 = [elem[1] for elem in X_train]
        ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=7)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(model.kernel + str(model.degree))

        xx = np.array(X_train)
        yy = np.array(y_train)
        xx = xx.astype(np.float64)
        yy = yy.astype(np.int32)
        X0, X1 = xx[:, 0], xx[:, 1]
        xx, yy = np.meshgrid(np.arange(X0.min() - 0.5, X0.max() + 0.5, .1),
                             np.arange(X1.min() - 0.5, X1.max() + 0.5, .1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()


task_a()
task_b()
task_cde("c")
task_cde("d")
task_cde("d", gamma="auto")
task_cde("e")
task_cde("e", gamma="auto")
task_cde("e", gamma=20)
task_cde("e", gamma=100)
task_cde("e", gamma=10000)
task_cde("e", gamma=10000, c=1000)
task_cde("e", gamma=100000)
