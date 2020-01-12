### Phân vùng ảnh (Image Segmentation)
import warnings
warnings.filterwarnings('ignore')

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MeanShift

# Phân vùng ảnh xám (GRAY)
img = cv2.imread('img1.jpg')
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)

# plt.imshow(img, cmap='gray')
# plt.show()

X = img.reshape((-1, 1))
print(X.shape)

# sử dụng thuật toán K-Means và MeanShift để phân vùng các điểm ảnh theo độ sáng của nó:
# K-Means
kmeans = KMeans(n_clusters=10, init='random').fit(X)
print(kmeans)
center = kmeans.cluster_centers_
label = kmeans.labels_

print(center)
print(label.shape)

segmented_image = center[label]
segmented_image = np.reshape(segmented_image, img.shape)

# plt.imshow(segmented_image, cmap='gray')
# plt.show()

# MeanShift
def init_seed(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

random_seed = init_seed(X, 10)
ms = MeanShift(bandwidth=1, seeds=random_seed)
print(ms)
ms.fit(X)
center = ms.cluster_centers_
label = ms.labels_

print(center)
print(label.shape)

segmented_image = center[label]
segmented_image = np.reshape(segmented_image, img.shape)
# plt.imshow(segmented_image/255.0, cmap='gray')
# plt.show()

# Phân vùng ảnh màu (RGB)
# - Sử dụng không gian đặc trưng 3D
img = plt.imread('img1.jpg')
print(img.shape)

# plt.imshow(img)
# plt.show()

def get_3D_vector(img):
     X = None
     X = img.reshape((-1, 3))
     return X

X = get_3D_vector(img)
print(X.shape)

# sử dụng thuật toán K-Means và MeanShift để phân vùng các điểm ảnh theo độ sáng của nó:
# K-Means
kmeans = KMeans(n_clusters=4, init='random').fit(X)
print(kmeans)
center = kmeans.cluster_centers_
label = kmeans.labels_

print(center)
print(label.shape)

segmented_image = center[label]
segmented_image = np.reshape(segmented_image, img.shape)

# plt.imshow(segmented_image/255.0)
# plt.show()

# MeanShift
random_seed = init_seed(X, 10)
ms = MeanShift(bandwidth=1, seeds=random_seed)
print(ms)
ms.fit(X)
center = ms.cluster_centers_
label = ms.labels_

print(center)
print(label.shape)

segmented_image = center[label]
segmented_image = np.reshape(segmented_image, img.shape)
# plt.imshow(segmented_image/255.0)
# plt.show()

# Sử dụng không gian đặc trưng 5D
img = plt.imread('img1.jpg')
print(img.shape)
# plt.imshow(img)
# plt.show()

def get_5D_vector(img):
    X = None
    X_pos = np.zeros((img.shape[0], img.shape[1], 2))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            X_pos[i][j][0] = i
            X_pos[i][j][1] = j
    X = img.reshape((-1, 3))
    X_pos = X_pos.reshape((-1, 2))
    X = np.concatenate((X, X_pos), axis=1)

    return X

X = get_5D_vector(img)
print(X.shape)

# sử dụng thuật toán K-Means và MeanShift để phân vùng các điểm ảnh theo độ sáng của nó:
# K-Means
kmeans = KMeans(n_clusters=10, init='random').fit(X)
print(kmeans)
center = kmeans.cluster_centers_
label = kmeans.labels_

print(center)
print(label.shape)

segmented_image = center[label]
segmented_image = segmented_image[:, :3]
segmented_image = np.reshape(segmented_image, img.shape)

plt.imshow(segmented_image/255.0)
plt.show()

# MeanShift
random_seed = init_seed(X, 10)
ms = MeanShift(bandwidth=10, seeds=random_seed)
print(ms)
ms.fit(X)
center = ms.cluster_centers_
label = ms.labels_

print(center)
print(label.shape)

segmented_image = center[label]
segmented_image = segmented_image[:, :3]
segmented_image = np.reshape(segmented_image, img.shape)
plt.imshow(segmented_image/255.0)
plt.show()


