import matplotlib.pyplot as mlt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=["sepal length","sepal width","petal length","petal width","class"])
iris.head()

X = iris.drop('class', axis=1).values
Y = iris['class'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

lr = LogisticRegression()

# folding with KFold
kfold = KFold(n_splits=10, shuffle=1)
scores = []

for k, (train, test) in enumerate(kfold.split(X_train)):
    lr.fit(X_train[train], Y_train[train])
    score = lr.score(X_train[test], Y_train[test])
    scores.append(score)

    print(f'Fold n {k}: Accuracy {score}')

mean_accuracy = np.array(scores).mean()
print(f'KFold - Final accuracy: {mean_accuracy}')

# folding with StratifiedKFold (equal n of examples for class)
kfold = StratifiedKFold(n_splits=10, shuffle=1)
scores = []

for k, (train, test) in enumerate(kfold.split(X_train, Y_train)):
    lr.fit(X_train[train], Y_train[train])
    score = lr.score(X_train[test], Y_train[test])
    scores.append(score)

    print(f'Fold n {k}: Accuracy {score}')

mean_accuracy = np.array(scores).mean()
print(f'StratifiedKFold - Final accuracy: {mean_accuracy}')

# KFold Cross Validation

lr = LogisticRegression()

score = cross_val_score(lr, X_train, Y_train, cv=10)

print(f'KFold Cross Val - Final accuracy: {score.mean()}')

lr.fit(X_train, Y_train)