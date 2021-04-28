import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

X1 = []
XY12 = []
XY11 = []
X2 = []
XY21 = []
XY22 = []
X3 = []
XY31 = []
XY32 = []
i = 0
for file in ("3/clustering_1.csv", "3/clustering_2.csv", "3/clustering_3.csv"):
    i += 1
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip('\n').split()
            if i == 1:
                X1.append(list(map(float, arr)))
                XY11.append(float(arr[0]))
                XY12.append(float(arr[1]))
            elif i == 2:
                X2.append(list(map(float, arr)))
                XY21.append(float(arr[0]))
                XY22.append(float(arr[1]))
            elif i == 3:
                X3.append(list(map(float, arr)))
                XY31.append(float(arr[0]))
                XY32.append(float(arr[1]))

#-------------1-st model-----------------------
print("1st model:")
plt.scatter(XY11, XY12)
plt.show()

print("K means")
kmeans = KMeans(n_clusters=2)
y1 = kmeans.fit_predict(X1)
print("n_iter = ", kmeans.n_iter_)
print("inertia = ",kmeans.inertia_)
plt.scatter(XY11, XY12, c=y1)
plt.show()

print('Hierarchical1')
hier = AgglomerativeClustering()
y1 = hier.fit_predict(X1)
print("n_leaves = ", hier.n_leaves_)
plt.scatter(XY11, XY12, c=y1)
plt.show()

print('Hierarchical2')
hier = AgglomerativeClustering(linkage="single")
y1 = hier.fit_predict(X1)
print("n_leaves = ", hier.n_leaves_)
plt.scatter(XY11, XY12, c=y1)
plt.show()

print("DBSCAN")
dbscan = DBSCAN()
y1 = dbscan.fit_predict(X1)
plt.scatter(XY11, XY12, c=y1)
plt.show()

print("--------------------------------------")
#-------------2-nd model-----------------------

print("2nd model:")
plt.scatter(XY21, XY22)
plt.show()

print("K means")
kmeans = KMeans(n_clusters=3)
y2 = kmeans.fit_predict(X2)
print("n_iter = ", kmeans.n_iter_)
print("inertia = ",kmeans.inertia_)
plt.scatter(XY21, XY22, c=y2)
plt.show()

print('Hierarchical1')
hier = AgglomerativeClustering(n_clusters=3)
y2 = hier.fit_predict(X2)
print("n_leaves = ", hier.n_leaves_)
plt.scatter(XY21, XY22, c=y2)
plt.show()

print('Hierarchical2')
hier = AgglomerativeClustering(n_clusters=3, linkage="single")
y2 = hier.fit_predict(X2)
print("n_leaves = ", hier.n_leaves_)
plt.scatter(XY21, XY22, c=y2)
plt.show()

print("DBSCAN")
dbscan = DBSCAN()
y2 = dbscan.fit_predict(X2)
plt.scatter(XY21, XY22, c=y2)
plt.show()
print("--------------------------------------")
#-------------3-rd model-----------------------

print("3rd model:")
plt.scatter(XY31, XY32)
plt.show()

print("K means")
kmeans = KMeans(n_clusters=6)
y3 = kmeans.fit_predict(X3)
print("n_iter = ", kmeans.n_iter_)
print("inertia = ",kmeans.inertia_)
plt.scatter(XY31, XY32, c=y3)
plt.show()

print('Hierarchical1')
hier = AgglomerativeClustering(n_clusters=6)
y3 = hier.fit_predict(X3)
print("n_leaves = ", hier.n_leaves_)
plt.scatter(XY31, XY32, c=y3)
plt.show()

print('Hierarchical2')
hier = AgglomerativeClustering(n_clusters=4, linkage="single")
y3 = hier.fit_predict(X3)
print("n_leaves = ", hier.n_leaves_)
plt.scatter(XY31, XY32, c=y3)
plt.show()

print("DBSCAN")
dbscan = DBSCAN()
y3 = dbscan.fit_predict(X3)
plt.scatter(XY31, XY32, c=y3)
plt.show()
print("--------------------------------------")