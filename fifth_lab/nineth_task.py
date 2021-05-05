import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data_nsw = pd.read_csv("5/nsw74psid1.csv")
data_nsw.describe()

X_train, X_test, y_train, y_test = train_test_split(data_nsw.drop(["re78"], axis=1), data_nsw["re78"], train_size=0.75)

models = [DecisionTreeRegressor(), LinearRegression(), SVR()]

for model in models:
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    model.score(X_train, y_train)
    print(model)
    print(f"test score: {model.score(X_test, y_test)}")
    print(f"train score: {model.score(X_train, y_train)}")
