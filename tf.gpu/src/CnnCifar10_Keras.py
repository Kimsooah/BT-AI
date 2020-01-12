print("Image Classification với tập dữ liệu CIFAR-10")

# Thu thập dữ liệu
print("Thu thập dữ liệu")
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

""" 
    Kiểm tra xem file dữ liệu (zip) được tải về chưa
    nếu chưa, tải về tại "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" với tên cifar-10-python.tar.gz
"""
if not isfile('cifar-10-python.tar.gz'):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

print("Một số nội dung trước khi bắt đầu với CIFAR-10")
import pickle
import numpy as np
import matplotlib.pyplot as plt

print("Nhãn của dữ liệu")
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Reshape về kích thước phù hợp cho CNN")
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

print("Khám phá dữ liệu")
def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))

    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

    plt.imshow(sample_image)
    plt.show()

import numpy as np

# Explore the dataset
batch_id = 3
sample_id = 7000
display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

# Cài đặt hàm tiền xử lý dữ liệu
print("Cài đặt hàm tiền xử lý dữ liệu")
print("Min-Max Normalization")
"""
* phương pháp này đơn giản chỉ thực hiện việc
đưa giá trị của x về khoảng giữa 0 và 1.
* y = (x-min) / (max-min)
"""
def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

print("One-hot encode")
def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded

print("Tiền xử lý dữ liệu")


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(normalize, one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation],
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         'preprocess_training.p')

preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

print("Checkpoint")
import pickle
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# Tạo mô hình Convolutional
print("Tạo mô hình Convolutional")
print("Lấy dữ liệu từ các file batch")
import pickle
import numpy as np

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(batch_id):
    features, labels = pickle.load(open('preprocess_batch_' + str(batch_id)+'.p', mode='rb'))
    return features, labels

# import data
x_valid, y_valid = pickle.load(open('preprocess_validation.p', mode='rb'))
x_test, y_test = pickle.load(open('preprocess_training.p', mode='rb'))
x_train, y_train = load_cfar10_batch(1)
y_train = np.array(y_train)

for i in np.arange(2,6):
    x_batch, y_batch = load_cfar10_batch(i)
    x_train = np.concatenate((x_train, x_batch), axis=0)
    y_train = np.concatenate((y_train, np.array(y_batch)), axis=0)

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)

# Build model with Keras
print("Build model with Keras")

# build model
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(256, kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
Dropout(0.25)
model.add(Dense(units=256, activation='relu'))
Dropout(0.25)
model.add(Dense(units=512, activation='relu'))
Dropout(0.25)
model.add(Dense(units=1024, activation='relu'))
Dropout(0.25)
model.add(Dense(units=10, activation='softmax'))

from keras.optimizers import SGD
# Compile model
epochs = 25
lr = 0.001
decay = lr/epochs
sgd = SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=epochs
          , validation_data=(x_valid, y_valid))

# save model as json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

print("Kiểm thử model đã lưu")
from keras.models import model_from_json

# load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("loaded model from disk")

# load test data
X, Y = pickle.load(open('preprocess_training.p', mode='rb'))
loaded_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y)

#accuracy
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

print("Dự đoán mô hình")
import matplotlib.pyplot as plt
classes = loaded_model.predict(X, batch_size=128)
preds = np.argmax(classes, axis=1)

for i in range(10):
    plt.imshow(X[i])
    predict = "Kết quả dự đoán: " + str(load_label_names()[preds[i]])
    output = "label:"+ str(load_label_names()[np.argmax(Y[i])])
    plt.text(0, -2, predict)
    plt.text(0, -4, output)
    plt.show()