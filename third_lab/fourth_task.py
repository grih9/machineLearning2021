# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.preprocessing import StandardScaler
# from scipy.cluster.hierarchy import dendrogram
#
# states = []
# years = []
# with open("3/votes.csv") as f:
#     lines = f.readlines()
#     years = lines[0].strip('\n').replace('"', '').split(",")
#     lines = lines[1:]
#     for line in lines:
#         arr = line.strip('\n').split(",")
#         for i in range(len(arr)):
#             try:
#                 arr[i] = float(arr[i])
#             except:
#                 arr[i] = 0
#         states.append(arr)
#
# print(states)
# hier = AgglomerativeClustering(distance_threshold=0, linkage="single", n_clusters=None)
# hier = hier.fit(states)
# print("n_leaves = ", hier.n_leaves_)
#
# states_num = [i for i in range(50)]
# print(len(hier.labels_))
# plt.figure(figsize=(15, 15))
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
#
# dendrogram(y1,
# labels=states_num,
# leaf_font_size=15)
#
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

data_votes = pd.read_csv('3/votes.csv')

data_votes.fillna(0, inplace=True)
mergings = linkage(data_votes, 'single')

states = [i for i in range(50)]

plt.figure(figsize=(15, 15))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

dendrogram(mergings,
labels=states,
leaf_font_size=15)

plt.show()