# -*- coding: utf-8 -*-

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import mglearn
import matplotlib.pyplot as plt
mglearn.plots.plot_knn_classification(n_neighbors=1)

# load dataset
loaddataset = load_digits()
# add noise data to original data
rng = np.random.RandomState(42)
noisefeatures = rng.normal(size = (len(loaddataset .data),50))
# features with noise
features_with_noise= np.hstack([loaddataset .data , noisefeatures])
# train test split
X_train, X_test, y_train, y_test = train_test_split(
features_with_noise, loaddataset.target, random_state=0, test_size=.2)
print('shape is ',X_train.shape)
select = SelectPercentile(percentile=50)
select.fit(X_train,y_train)
mask = select.get_support()
print(mask)
X_train.shape
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)
X_train_selected.shape
# test logistic reg on dataset before feature selection
l=KNeighborsClassifier()
l.fit(X_train,y_train)
print('train accuracy',l.score(X_train,y_train))
# test logistic reg on dataset before feature selection
l.fit(X_train_selected,y_train)
print('test accuracy',l.score(X_test_selected,y_test))

print(load_digits().data.shape)
(1797, 64)
plt.gray()
plt.matshow(load_digits().images[1])
plt.show( 