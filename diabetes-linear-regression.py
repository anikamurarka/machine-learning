from sklearn import datasets
diabetes = datasets.load_diabetes()
type(data)
x = diabetes.data
y = diabetes.target

type(x)
x.shape
y.shape

import pandas as pd
df = pd.DataFrame(x)
df.columns = diabetes.feature_names
df

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x,y)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

from sklearn.linear_model import LinearRegression
alg1 = LinearRegression()
alg1.fit(X_train, Y_train)
alg1.fit(X_train, Y_train)
Y_pred = alg1.predict(X_test)

import matplotlib.pyplot as plt
import numpy as np
plt.scatter(Y_test,Y_pred)
a = np.array((0,50,200,350))
b =a
plt.plot(a,b)
plt.axis([30,350,30,350])
plt.show()
