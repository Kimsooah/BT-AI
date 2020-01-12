# Kiểm thử model đã lưu
# load model
import pickle
import numpy as np
from keras.models import model_from_json
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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

# Dự đoán mô hình
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