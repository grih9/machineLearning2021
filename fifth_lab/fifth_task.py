import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("5/eustock.csv", sep=',')
data.plot()
plt.title("Ежедневные котировки")
plt.ylabel("Котировки")
plt.legend()
plt.show()

X_train = np.array(data.index).reshape(-1, 1)

for y_train in [data['DAX'], data['SMI'], data['CAC'], data['FTSE']]:
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_train)
    plt.title(y_train.name)
    plt.plot(X_train, pred)
    plt.plot(X_train, y_train)
    plt.show()

for y_train in [data["DAX"], data["SMI"], data["CAC"], data["FTSE"]]:
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_train)
    plt.plot(X_train, pred, label=y_train.name)

plt.title("Линейные регрессии для котировок")
plt.ylabel("Котировки")
plt.legend(loc="best")
plt.show()

