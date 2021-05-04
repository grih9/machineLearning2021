import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression

data = pd.read_csv("5/reglab.txt", sep='\t')

for k in range(1, 5):
    for elem in list(combinations(["x1", "x2", "x3", "x4"],  k)):
        X_train = data[list(elem)]
        y_train = data["y"]
        model = LinearRegression()
        model.fit(X_train, y_train)
        print(f"y({elem}): {model.score(X_train, y_train)}")
