import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression

data = pd.read_csv("5/reglab1.txt", sep='\t')

X_train = data[["y", "z"]]
y_train = data["x"]
model = LinearRegression()
model.fit(X_train, y_train)
print(f"x(y, z): {model.score(X_train, y_train)}")

X_train = data[["z", "x"]]
y_train = data["y"]
model = LinearRegression()
model.fit(X_train, y_train)
print(f"y(x, z): {model.score(X_train, y_train)}")

X_train = data[["x", "y"]]
y_train = data["z"]
model = LinearRegression()
model.fit(X_train, y_train)
print(f"z(x, y): {model.score(X_train, y_train)}")
