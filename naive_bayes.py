# Naive Bayes example
from sklearn.naive_bayes import GaussianNB
import numpy as np

# assigning features and labels
X = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])


classifier = GaussianNB()

classifier.fit(X, y)



# Predict
new_feature = [[1, 2], [3,4]]
new_label = classifier.predict(new_feature)

print(new_label)

