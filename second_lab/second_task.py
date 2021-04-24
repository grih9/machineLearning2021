from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pylab as plt


X = list()
y = list()

with open("2/nn_1.csv", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        arr = line.strip('\n').split(",")
        X.append(list(map(float, arr[:2])))
        y.append(int(arr[2]))

X00 = [X[i][0] for i in range(len(X)) if y[i] == 1]
X10 = [X[i][1] for i in range(len(X)) if y[i] == 1]
X01= [X[i][0] for i in range(len(X)) if y[i] == -1]
X11 = [X[i][1] for i in range(len(X)) if y[i] == -1]
plt.scatter(X00, X10, label="one class")
plt.scatter(X01, X11, label="minus one class")
plt.legend(loc="best")
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

for activation_func in ("relu", "identity", "logistic", "tanh"):
    for solver in ("lbfgs", "sgd", "adam"):
        for random_state in (None, 1):
            mlp = MLPClassifier(random_state=random_state, hidden_layer_sizes=(),
                                max_iter=10000, activation=activation_func, solver=solver)
            mlp.fit(X_train, y_train)
            print("------------------------")
            print("activation:", activation_func + ", solver:", solver + ", random_state:", random_state)
            print("n_iter:", mlp.n_iter_)
            try:
                print("best_loss:", mlp.best_loss_)
            except:
                print("best_loss: None")
            pred = mlp.predict(X_test)
            print("test accuracy:", end=" ")
            print(metrics.accuracy_score(y_test, pred))
            pred = mlp.predict(X_train)
            print("train accuracy:", end=" ")
            print(metrics.accuracy_score(y_train, pred))
print("----------------------------------")
print("----------------------------------")
print("----------------------------------")
print("ADD HIDDEN LAYER")
print("----------------------------------")
print("----------------------------------")
print("----------------------------------")
for activation_func in ("relu", "identity", "logistic", "tanh"):
    for solver in ("lbfgs", "sgd", "adam"):
        for random_state in (None, 1):
            mlp = MLPClassifier(random_state=random_state, hidden_layer_sizes=(10),
                                max_iter=10000, activation=activation_func, solver=solver)
            mlp.fit(X_train, y_train)
            print("------------------------")
            print("activation:", activation_func + ", solver:", solver + ", random_state:", random_state)
            print("n_iter:", mlp.n_iter_)
            try:
                print("best_loss:", mlp.best_loss_)
            except:
                print("best_loss: None")
            pred = mlp.predict(X_test)
            print("test accuracy:", end=" ")
            print(metrics.accuracy_score(y_test, pred))
            pred = mlp.predict(X_train)
            print("train accuracy:", end=" ")
            print(metrics.accuracy_score(y_train, pred))