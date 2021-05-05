import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

data = pd.read_csv("5/longley.csv", sep=',')
data.drop("Population", axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data.drop(["Employed"], axis=1), data["Employed"], train_size=0.5)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("linear test", metrics.mean_squared_error(y_test, pred))
pred = model.predict(X_train)
print("linear train", metrics.mean_squared_error(y_train, pred))

r_test = []
r_train = []
params = [10 ** (-3 + 0.2 * i) for i in range(26)]
for param in params:
    model = Ridge(alpha=param)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r_test.append(metrics.mean_squared_error(y_test, pred))
    pred = model.predict(X_train)
    r_train.append(metrics.mean_squared_error(y_train, pred))

plt.plot([i for i in range(26)], r_test, label="test")
plt.plot([i for i in range(26)], r_train, label="train")
plt.legend()
plt.show()
plt.plot(params, r_test, label="test")
plt.plot(params, r_train, label="train")
plt.legend()
plt.show()
