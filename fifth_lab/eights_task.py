import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVR

data = pd.read_csv('5/svmdata6.txt', sep='\t')
X_train = np.array(data['X']).reshape(-1, 1)
y_train = data['Y']

epsilons = [i / 100 for i in range(150)]
error = []

for epsilon in epsilons:
    model = SVR(C=1.0, epsilon=epsilon)
    model.fit(X_train, y_train)

    pred = model.predict(X_train)
    error.append(metrics.mean_squared_error(y_train, pred))

plt.plot(epsilons, error, label='test')
plt.xlabel("epsilon")
plt.ylabel("error")
plt.show()
