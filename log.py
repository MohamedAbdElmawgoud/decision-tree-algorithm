# -*- coding: utf-8 -*-

# using  simple example for implmenting classfier on simple dataset
from sklearn.linear_model import LogisticRegression
Studied = [4.85,8.62,5.34,9.21]
Slept = [9.36,3.23,8.23,6.34]
passed = [1,0,1,0]
features=list(zip(Studied,Slept))
model = LogisticRegression()
model.fit(features,passed)
res=model.predict([[5.34,8.23]])
#print(le.inverse_transform(res))
#print(features)

# using loaddataset dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
loaddataset = load_digits()
x_train , x_test , y_tarin , y_test = train_test_split(
      loaddataset.data , loaddataset.target , random_state=0, test_size=.3  )
md=LogisticRegression(C=100 , penalty="l1")
md.fit(x_train,y_tarin)
print("training data", md.score(x_train,y_tarin))
print("test data",md.score(x_test,y_test))
md.intercept_
md.coef_
md.decision_function
# using
from sklearn.metrics  import classification_report
y_pred = md.predict(x_test)
#_print(y_pred)
#print(x_test)
classification_report(y_test,y_pred)
print(load_digits().data.shape)
(1797, 64)
plt.gray()
plt.matshow(load_digits().images[1])
plt.show()