import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)

train.shape
test.shape

train = train.drop(['ID'], axis=1)
testid = test.ID
test = test.drop(['ID'], axis=1)
target = train.target
train = train.drop(['target'], axis=1)


train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)


delcol = []
trainColumns = train.columns
for col in trainColumns:
    if train[col].isnull().sum() > 100000:
        delcol.append(col)
train = train.drop(delcol, axis = 1)
test = test.drop(delcol, axis = 1)


trainColumns = train.select_dtypes(['object']).columns
delcol = []
for col in trainColumns:
    if ((train[col] == '-1').sum() + train[col].isnull().sum()) > 100000:
        delcol.append(col)
train = train.drop(delcol, axis = 1)
test = test.drop(delcol, axis = 1)

#check what is contained
#trainColumns = train.select_dtypes(['object']).columns
#for col in trainColumns:
#    train[col].unique()
#    train[col].head(1)


#dates columns left
trainColumns = ['VAR_0075', 'VAR_0204', 'VAR_0217']
for col in trainColumns:
    train[col] = pd.to_datetime(train[col], format='%d%b%y:%H:%M:%S')
    test[col] = pd.to_datetime(test[col], format='%d%b%y:%H:%M:%S')

for col in trainColumns:
    train[col] = (train[col] - train[col].min()).astype('timedelta64[D]')
    test[col] = (test[col] - test[col].min()).astype('timedelta64[D]')


#booleans columns
trainColumns = ['VAR_0008', 'VAR_0009', 'VAR_0010', 'VAR_0011', 'VAR_0012', 'VAR_0043', 'VAR_0196', 'VAR_0226', 'VAR_0229', 'VAR_0230', 'VAR_0232', 'VAR_0236', 'VAR_0239']
train[trainColumns] = train[trainColumns].fillna(train[trainColumns].median()).astype(int)
test[trainColumns] = test[trainColumns].fillna(train[trainColumns].median()).astype(int)


trainColumns = train.blocks['object'].columns
train[trainColumns] = train[trainColumns].fillna('-1')
test[trainColumns] = test[trainColumns].fillna('-1')


#failed in label encoder
train = train.drop(['VAR_0200'], axis=1)
test = test.drop(['VAR_0200'], axis=1)

test['VAR_0237'] = test['VAR_0237'].replace('ND', train['VAR_0237'].mode().astype(str).iloc[0])
test['VAR_0283'] = test['VAR_0283'].replace('G', train['VAR_0283'].mode().astype(str).iloc[0])

from sklearn import preprocessing
trainColumns = train.select_dtypes(['object']).columns
for col in trainColumns:
    le = preprocessing.LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


for col in train.columns:
    x = train[col].median()
    if train[col].isnull().sum() > 0:
        train[col] = train[col].fillna(x)
    if test	[col].isnull().sum() > 0:
        test[col] = test[col].fillna(x)


from sklearn import preprocessing
for col in train.columns:
    scaler = preprocessing.StandardScaler()
    train[col] = scaler.fit_transform(train[col])
    test[col] = scaler.transform(test[col])


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500, random_state = 12345)
forest = forest.fit(train, target)
output = forest.predict(test)


df = pd.DataFrame()
df["ID"] = testid
df["target"] = output
df.to_csv('output1.csv', index = False)





#strings
#['VAR_0001', 'VAR_0005', 'VAR_0044', 'VAR_0200', 'VAR_0202', 'VAR_0216', 'VAR_0222', 'VAR_0237', 'VAR_0274', 'VAR_0283', 'VAR_0305', 'VAR_0325', 'VAR_0342', 'VAR_0352', 'VAR_0353', 'VAR_0354', 'VAR_1934']

#dates
#['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204', 'VAR_0217']


#delcol = ['VAR_0008', 'VAR_0009', 'VAR_0010', 'VAR_0011', 'VAR_0012', 'VAR_0043', 'VAR_0044', 'VAR_0196', 'VAR_0216', 'VAR_0222', 'VAR_0229', 'VAR_0239']
#for col in delcol:
#    train = train.drop([col], axis = 1)
#    test = test.drop([col], axis = 1)

#train = train.dropna(axis=0,how='any',subset=['VAR_0075'])
#train = train.dropna(axis=0,how='any',subset=['VAR_0274'])
#train = train.dropna(axis=0,how='any',subset=['VAR_0111'])
