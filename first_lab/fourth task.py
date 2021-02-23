def task_a():
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
    print(metrics.accuracy_score(pred, y_test))
    print(metrics.confusion_matrix(y_test, pred))
    pred = clf.predict(X_train)
    print(metrics.accuracy_score(pred, y_train))
    print(metrics.confusion_matrix(y_train, pred))
    print(clf.support_vectors_)
    print(clf.n_support_)
    X0 = [elem[0] for elem in X_train]
    X1 = [elem[1] for elem in X_train]
    plt.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=70)
    X0t = [elem[0] for elem in X_test]
    X1t = [elem[1] for elem in X_test]
    plt.scatter(X0t, X1t, c=y_train, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')

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

task_a()
