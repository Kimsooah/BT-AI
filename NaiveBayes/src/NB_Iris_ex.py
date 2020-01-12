print("Vi du ve bai toan phan lop hoa Iris bang NaiveBayes")

from sklearn import datasets, metrics
from sklearn.naive_bayes import GaussianNB

# load  the Iris datasets
dataset = datasets.load_iris()
print(dataset.data[0:6])

expected = dataset.target[0:100]
print(expected)

# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
print(expected)

predicted = model.predict(dataset.data)
print(predicted)

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))