import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('5/cars.csv')

plt.scatter(data["speed"], data["dist"])
X_train = np.array(data["speed"]).reshape(-1, 1)
y_train = data["dist"]
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_train)
plt.plot(X_train,pred)
plt.xlabel("Скорость")
plt.ylabel("Длина тормозного пути")
plt.show()
X_test = [[40]]
predicted = model.predict(X_test)
print(f"f(40): {predicted[0]}")
