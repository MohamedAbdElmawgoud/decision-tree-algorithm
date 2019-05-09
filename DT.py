## Part1

import mglearn # credits to Muller and Guido (https://www.amazon.com/dp/1449369413/)
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
mglearn.plots.plot_tree_not_monotone()

## Part 2
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

loaddataset = load_digits()
# add noise data to original data
rng = np.random.RandomState(42)
noisefeatures = rng.normal(size = (len(loaddataset .data),50))
# features with noise
features_with_noise= np.hstack([loaddataset .data , noisefeatures])
# train test split
X_train, X_test, y_train, y_test = train_test_split(
features_with_noise, loaddataset.target, random_state=0, test_size=.2)
tree = DecisionTreeClassifier(random_state=1)

tree.fit(X_train, y_train)
print('Feature importances: {}'.format(tree.feature_importances_))
type(tree.feature_importances_)
print('\nAccuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=13, random_state=1)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))

## Part 3

print(load_digits().data.shape)
(1797, 64)
plt.gray()
plt.matshow(load_digits().images[1])
plt.show()


