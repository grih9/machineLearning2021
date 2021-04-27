import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = []

with open("3/pluton.csv") as f:
    lines = f.readlines()
    headers = lines[0].strip('\n').replace('"', '').split(",")
    print(headers)
    lines = lines[1:]
    for line in lines:
        arr = line.strip('\n').split(",")
        X.append(list(map(float, arr)))

scaledX = StandardScaler().fit_transform(X)
for max_iter in range(1, 15):
    for random_state in (None, 1):
        kmeans = KMeans(n_clusters=3, max_iter=max_iter, random_state=random_state)
        kmeansScaled = KMeans(n_clusters=3, max_iter=max_iter, random_state=random_state)
        kmeans.fit(X)
        kmeansScaled.fit(scaledX)
        print(f"max_iter={max_iter}, random_state={random_state}")
        print("obvious n_iter = ", kmeans.n_iter_)
        print("obvious inertia = ",kmeans.inertia_)
        print("scaled n_iter = ", kmeansScaled.n_iter_)
        print("scaled inertia = ", kmeansScaled.inertia_)
