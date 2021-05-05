import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("5/JohnsonJohnson.csv")
new_data = data["index"].str.split(' ', expand=True)
new_data.columns = ["year", "quarter"]

data = pd.concat([data, new_data], axis=1)

data.drop(["index"], axis=1, inplace=True)
print(data.head())
q1 = data[data["quarter"] == "Q1"]
q2 = data[data["quarter"] == "Q2"]
q3 = data[data["quarter"] == "Q3"]
q4 = data[data["quarter"] == "Q4"]

plt.plot(q1.year, q1.value, label="Q1")
plt.plot(q2.year, q2.value, label='Q2')
plt.plot(q3.year, q3.value, label='Q3')
plt.plot(q4.year, q4.value, label='Q4')

plt.title("Изменения прибыли")
plt.xlabel("year")
plt.ylabel("value")
plt.xticks(rotation=90)
plt.legend()
plt.show()

y_train_list = [q1["value"], q2["value"], q3["value"], q4["value"]]
X_train = np.array(q1.year).reshape(-1, 1)

for i, y_train in enumerate(y_train_list):
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_train)

    plt.title(f"Q{i + 1}")
    plt.plot(q1.year, pred)
    plt.plot(q1.year, y_train)
    plt.xlabel("year")
    plt.ylabel("value")
    plt.xticks(rotation=90)
    plt.show()

for i, y_train in enumerate(y_train_list):
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_train)
    plt.plot(q1.year, pred, label=f"Q{i + 1}")

plt.xlabel("year")
plt.ylabel("value")
plt.legend()
plt.xticks(rotation=90)
plt.show()

reg_data = data

label_encoder = LabelEncoder()
reg_data["quarter"] = label_encoder.fit_transform(data["quarter"])
reg_data["year_month"] = reg_data["year"].astype(int) + ((reg_data["quarter"] + 1) / 4)

X_train_reg = np.array(reg_data["year_month"]).reshape(-1, 1)
y_train_reg = reg_data["value"]

model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)

pred = model_reg.predict(X_train_reg)

plt.title("Линейная регрессия")
plt.plot(reg_data["year_month"], y_train_reg)
plt.plot(reg_data["year_month"], pred)
plt.xticks(rotation=90)
plt.show()

X_test = [[2016]]

for i, y_train in enumerate(y_train_list):
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Q{i + 1}:", pred[0])

pred = model_reg.predict(X_test)
print("В среднем:", pred[0])