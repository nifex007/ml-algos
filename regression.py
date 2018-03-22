# Regression algorithm in Stock market prediction 
# Goal: to predict a future price of stock from old stock prices from previous
import quandl, math
import pandas
import numpy as np

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot') #to display



df = quandl.get('WIKI/GOOGL')

# print(df.head())

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


# print(df.head())


forecast_col = 'Adj. Close'
df.fillna(value=-9999, inplace=True) #default input for NaN datapoint
forecast_out = int(math.ceil(0.01 * len(df))) #days ahead


#add a new column
df['label'] = df[forecast_col].shift(-forecast_out)





#define features as X and labels and y
X = np.array(df.drop(['label'], 1)) #remove lable to remain only features


#Scaling data with preprocessing to reduce errors (-1,1) 
X = preprocessing.scale(X)

X = X[:-forecast_out+1] #forcast the stock prices for one day ahead

df.dropna(inplace=True)
y=np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) #test 20% of the data


# Using a classifer
classifier = LinearRegression(n_jobs=-1) # 97%
# classifier = svm.SVR() #81%

classifier.fit(X_train, y_train) 	#Train
confidence = classifier.score(X_test, y_test) 	#Test

forecast_set = classifer.predict(X_lately)
df['Forecast'] = np.nan 

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400

next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()











