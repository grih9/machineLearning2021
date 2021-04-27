import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

flower = load_sample_image("flower.jpg")
flower = np.array(flower, dtype=np.float64) / 255
plt.figure()
plt.clf()
plt.axis('off')
plt.title('Original')
plt.imshow(flower)

for n_colors in (128, 64, 32, 16, 8, 4):
    flower = load_sample_image("flower.jpg")
    flower = np.array(flower, dtype=np.float64) / 255

    w, h, d = original_shape = tuple(flower.shape)
    image_ar = np.reshape(flower, (w * h, d))

    image_reduced = shuffle(image_ar, random_state=0)[:3000]
    kmeans = KMeans(n_clusters=n_colors).fit(image_reduced)
    labels = kmeans.predict(image_ar)

    plt.figure()
    plt.clf()
    plt.axis('off')
    plt.title(f'{n_colors} colors')
    image = np.zeros((w, h, 3))
    ind = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = kmeans.cluster_centers_[labels[ind]]
            ind += 1
    plt.imshow(image)
    plt.show()
