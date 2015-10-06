import pandas as pd
import numpy as np
import datetime

train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)

train = train.drop(['ID'], axis=1)
testid = test.ID
test = test.drop(['ID'], axis=1)
target = train.target
train = train.drop(['target'], axis=1)


#taking care of date columns
dateColumns = ['VAR_0073','VAR_0075','VAR_0156','VAR_0157','VAR_0158','VAR_0159','VAR_0166','VAR_0167','VAR_0168','VAR_0169','VAR_0176','VAR_0177','VAR_0178','VAR_0179','VAR_0204','VAR_0217']
for col in dateColumns:
    train[col] = pd.to_datetime(train[col], format='%d%b%y:%H:%M:%S')
    test[col] = pd.to_datetime(test[col], format='%d%b%y:%H:%M:%S')

for col in dateColumns:
    mindate = train[col].min()
    train[col] = (train[col] - mindate).astype('timedelta64[D]')
    test[col] = (test[col] - mindate).astype('timedelta64[D]')


#taking care of bool columns
boolColumns = ['VAR_0008','VAR_0009','VAR_0010','VAR_0011','VAR_0012','VAR_0043','VAR_0196','VAR_0226','VAR_0229','VAR_0230','VAR_0232','VAR_0236','VAR_0239']
train[boolColumns] = train[boolColumns].fillna(-1).astype(int)
test[boolColumns] = test[boolColumns].fillna(-1).astype(int)

#taking care of the rest
trainColumns = train.blocks['object'].columns
train[trainColumns] = train[trainColumns].fillna('-1')
test[trainColumns] = test[trainColumns].fillna('-1')

train = train.fillna(-1)
test = test.fillna(-1)

from sklearn import preprocessing
trainColumns = train.select_dtypes(['object']).columns
for col in trainColumns:
    le = preprocessing.LabelEncoder()
    le.fit(train[col].append(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500, random_state = 1104)
forest = forest.fit(train, target)
probs = forest.predict_proba(test)
output = ["%f" % x[1] for x in probs]


df = pd.DataFrame()
df["ID"] = testid
df["target"] = output
df.to_csv('output.csv', index = False)
