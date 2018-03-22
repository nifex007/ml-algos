# K Nearest Neighbors 

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) #replace empty data by -99999
df.drop(['id'], 1, inplace=True) #drop the id column

# everything is a feature except for the class column
X = np.array(df.drop(['class'], 1)) #features
y = np.array(df['class'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,test_size=0.2)
classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(accuracy)