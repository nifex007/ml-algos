# Naive Bayes example
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
import numpy as np

# assigning features and labels
X = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

# model
classifier = GaussianNB()

#train
classifier.fit(X, y)
# check confidence or accuracy of the model
print(classifier.score(X, y)) 

# Predict
new_feature = [[1, 2], [3,4]]
new_label = classifier.predict(new_feature)

print(new_label)

