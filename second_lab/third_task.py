from keras.datasets import mnist
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pylab as plt

X = list()
y = list()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  ' + str(X_test.shape))
print('Y_test:  ' + str(y_test.shape))

X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  ' + str(X_test.shape))
print('Y_test:  ' + str(y_test.shape))

mlp = MLPClassifier(random_state=1, hidden_layer_sizes=(100, 100, 100),
                    max_iter=1000000)
mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)
print(mlp.n_iter_)
print(metrics.accuracy_score(y_test, pred))

mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100),
                    max_iter=1000000)
mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)
print(mlp.n_iter_)
print(metrics.accuracy_score(y_test, pred))

# mlp = MLPClassifier(random_state=1, hidden_layer_sizes=(50, 40, 30),
#                     max_iter=1000000)
# mlp.fit(X_train, y_train)
# pred = mlp.predict(X_test)
# print(mlp.n_iter_)
# print(metrics.accuracy_score(y_test, pred))
#
# mlp = MLPClassifier(hidden_layer_sizes=(50, 40, 30),
#                     max_iter=1000000)
# mlp.fit(X_train, y_train)
# pred = mlp.predict(X_test)
# print(mlp.n_iter_)
# print(metrics.accuracy_score(y_test, pred))
#
# mlp = MLPClassifier(random_state=1, hidden_layer_sizes=(50),
#                     max_iter=1000000)
# mlp.fit(X_train, y_train)
# pred = mlp.predict(X_test)
# print(mlp.n_iter_)
# print(metrics.accuracy_score(y_test, pred))
#
# mlp = MLPClassifier(hidden_layer_sizes=(50),
#                     max_iter=1000000)
# mlp.fit(X_train, y_train)
# pred = mlp.predict(X_test)
# print(mlp.n_iter_)
# print(metrics.accuracy_score(y_test, pred))
#
# mlp = MLPClassifier(random_state=1, hidden_layer_sizes=(150),
#                     max_iter=1000000)
# mlp.fit(X_train, y_train)
# pred = mlp.predict(X_test)
# print(mlp.n_iter_)
# print(metrics.accuracy_score(y_test, pred))
#
# mlp = MLPClassifier(hidden_layer_sizes=(150),
#                     max_iter=1000000)
# mlp.fit(X_train, y_train)
# pred = mlp.predict(X_test)
# print(mlp.n_iter_)
# print(metrics.accuracy_score(y_test, pred))

mlp = MLPClassifier(random_state=1, hidden_layer_sizes=(350),
                    max_iter=1000000)
mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)
print(mlp.n_iter_)
print(metrics.accuracy_score(y_test, pred))

mlp = MLPClassifier(hidden_layer_sizes=(350),
                    max_iter=1000000)
mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)
print(mlp.n_iter_)
print(metrics.accuracy_score(y_test, pred))

mlp = MLPClassifier(random_state=1, hidden_layer_sizes=(),
                    max_iter=1000000)
mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)
print(mlp.n_iter_)
print(metrics.accuracy_score(y_test, pred))