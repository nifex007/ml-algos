# Regression algorithm in Stock market prediction 
# Goal: to predict a future price of stock from old stock prices from previous
import quandl, math
import pandas
import numpy as np

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression



df = quandl.get('WIKI/GOOGL')

# print(df.head())

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


print(df.head())


forecast_col = 'Adj. Close'
df.fillna(value=-9999, inplace=True) #default input for NaN datapoint
forecast_out = int(math.ceil(0.01 * len(df))) #days ahead


#add a new column
df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

#define features as X and labels and y
X = np.array(df.drop(['label'], 1)) #remove lable to remain only features
y = np.array(df['label']) 

#Scaling data with preprocessing to reduce errors (-1,1) 
X = preprocessing.scale(X)

# X = X[:-forecast_out+1] #forcast the stock prices for one day ahead

# df.dropna(inplace=True)
y=np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) #test 20% of the data


# Using a classifer
classifier = LinearRegression() # 97%
# classifier = svm.SVR() #81%

classifier.fit(X_train, y_train) 	#Train
confidence = classifier.score(X_test, y_test) 	#Test

print(confidence)









