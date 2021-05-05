import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("5/cygage.txt", sep='\t')

plt.scatter(data["Depth"], data["calAge"], label="data")

X_train = np.array(data["Depth"]).reshape(-1, 1)
y_train = data["calAge"]
weights = data["Weight"]
model = LinearRegression()
model.fit(X_train, y_train, sample_weight=weights)
print(f"calAge(Depth): {model.score(X_train, y_train, sample_weight=weights)}")
predicted = model.predict(X_train)
plt.plot(X_train, predicted, label="regression")
plt.show()