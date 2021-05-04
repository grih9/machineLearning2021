import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("5/cygage.txt", sep='\t')

X_train = np.array(data["Depth"]).reshape(-1, 1)
y_train = data["calAge"]
weights = data["Weight"]
model = LinearRegression()
model.fit(X_train, y_train, sample_weight=weights)
print(f"calAge(Depth): {model.score(X_train, y_train, sample_weight=weights)}")
