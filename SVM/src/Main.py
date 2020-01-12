### Phân loại phương tiện giao thông
# Bi thieu thu muc: trainingset
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

### Khảo sát tập dữ liệu
def load_image(image_path):
    return cv2.imread(image_path)

# Quan sát một vài mẫu dữ liệu
samples_list = []
samples_label = []
for label in os.listdir('trainingset'):
    sample_file = os.listdir(os.path.join('trainingset', label))[0]
    samples_list.append(load_image(os.path.join('trainingset', label, sample_file)))
    samples_label.append(label)

for i in range(len(samples_list)):
    plt.subplot(2, 3, i + 1), plt.imshow(cv2.cvtColor(samples_list[i], cv2.COLOR_BGR2RGB))
    plt.title(samples_label[i]), plt.xticks([]), plt.yticks([])

plt.show()

# Thống kê dữ liệu
def statistic():
    label = []
    num_images = []
    for lab in os.listdir('trainingset'):
        label.append(lab)
        num_images.append(len(os.listdir(os.path.join('trainingset', lab))))

    return label, num_images

label, num_images = statistic()
y_pos = np.arange(len(label))
plt.barh(y_pos, num_images, align='center', alpha=0.5)
plt.yticks(y_pos, label)
plt.show()

print('Total images: %d' %(sum(num_images)))

# Xây dựng danh sách chứa ảnh
def read_data(label2id):
    X = []
    Y = []

    for label in os.listdir('trainingset'):
        for img_file in os.listdir(os.path.join('trainingset', label)):
            img = load_image(os.path.join('trainingset', label, img_file))
            X.append(img)
            Y.append(label2id[label])
    return X, Y

# Label to id, used to convert string label to integer
label2id = {'pedestrian':0, 'moto':1, 'truck':2, 'car':3, 'bus':4}
X, Y = read_data(label2id)
print(len(X))
print(len(Y))
for i in range(10):
    print(X[i].shape)
print(Y[0])

### Trích xuất đặc trưng (features extraction)
# Trích xuất đặc trưng SIFT
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.xfeatures2d.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors

image_descriptors = extract_sift_features(X)
print(len(image_descriptors))
print(type(image_descriptors[0][0]))
for i in range(10):
    print('Image {} has {} descriptors'.format(i, len(image_descriptors[i])))

# Xây dựng từ điển
all_descriptors = []
for descriptors in image_descriptors:
    if descriptors is not None:
        for des in descriptors:
            all_descriptors.append(des)
print('Total number of descriptors: %d' %(len(all_descriptors)))

def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_

    return bow_dict


num_clusters = 150
if not os.path.isfile('bow_dictionary150.pkl'):
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump(BoW, open('bow_dictionary150.pkl', 'wb'))
else:
    BoW = pickle.load(open('bow_dictionary150.pkl', 'rb'))

print(len(BoW))
print(type(BoW[0]))

# Xây dựng vector đặc trưng với mô hình BoW
from scipy.spatial.distance import cdist
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []

    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)
        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)

    return X_features

print(image_descriptors[0].shape)
print(BoW[0].shape)
X_features = create_features_bow(image_descriptors, BoW, num_clusters)
print(len(X_features))
print(X_features[0])
print(sum(X_features[0]))
print(image_descriptors[0].shape[0])

### Xây dựng mô hình
print(len(X_features))
print(len(Y))

# Chia tập dữ liệu thành tập train/test:
from sklearn.model_selection import train_test_split
X_train = []
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)
print(len(X_train))
print(len(X_test))

svm = sklearn.svm.SVC(C = 30, max_iter=100)
print(svm)

# Huấn luyện mô hình:
svm.fit(X_train, Y_train)

# Tính độ chính xác
print(svm.score(X_train, Y_train))
print(svm.score(X_test, Y_test))

# Lựa chọn tham số:
from sklearn.model_selection import GridSearchCV
svm = sklearn.svm.SVC()
param = {'C': [30, 50, 100], 'kernel': ['rbf', 'linear'], 'class_weight': [None, 'balanced']}
gs = GridSearchCV(estimator=svm, param_grid=param, cv=3, n_jobs=-1)
print(gs)
gs.fit(X_train, Y_train)
gs.score(X_test, Y_test)

print(gs.best_params_)

### Dùng mô hình đã huấn luyện dự đoán hình ảnh thực tế
# Bước 1: Đọc ảnh ở đường dẫn image_test/car.png, lưu ảnh vào biến img
img = None
img = load_image('image_test/car.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Bước 2: Trích xuất đặc trưng SIFT (lưu vào biến my_image_descriptors) và BoW (lưu vào biến my_X_features) từ my_X:
my_X = [img]
my_image_descriptors = None
my_X_features = None
my_image_descriptors = extract_sift_features(my_X)
my_X_features = create_features_bow(my_image_descriptors, BoW, num_clusters)

print(len(my_image_descriptors))
print(my_X_features[0].shape)

# Bước 3: Sử dụng mô hình đã huấn luyện để dự đoán, kết quả dự đoán lưu vào biến my_y_pred
y_pred = None
y_pred = svm.predict(my_X_features)

print(y_pred)
print(label2id)
for key, value in label2id.items():
    if value == y_pred[0]:
        print('Your prediction: ', key)














