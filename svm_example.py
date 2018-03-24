# Support Vector Machine

import numpy as np
from sklearn import preprocessing, cross_validation, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt') #data source

df.replace('?', -99999, inplace=True) #replace empty data by -99999
df.drop(['id'], 1, inplace=True) #drop the id column

# everything is a feature except for the class column
X = np.array(df.drop(['class'], 1)) #features
y = np.array(df['class'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,test_size=0.2)

classifier = svm.SVC() #using Support vector machine

classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(accuracy)

# #  Attribute                     Domain
#    -- -----------------------------------------
#    1. Sample code number            id number //this column is dropped and not passed along side args into np.array()
#    2. Clump Thickness               1 - 10
#    3. Uniformity of Cell Size       1 - 10
#    4. Uniformity of Cell Shape      1 - 10
#    5. Marginal Adhesion             1 - 10
#    6. Single Epithelial Cell Size   1 - 10
#    7. Bare Nuclei                   1 - 10
#    8. Bland Chromatin               1 - 10
#    9. Normal Nucleoli               1 - 10
#   10. Mitoses                       1 - 10

# predict either 2. benign or 4. malignant
example_measures = np.array([2,3,4,5,6,4,3,5,6])
example_measures = example_measures.reshape(1,-1)
prediction = classifier.predict(example_measures)
print(prediction)

