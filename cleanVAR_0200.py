import pandas as pd
import numpy as np
from fuzzywuzzy import process

train = pd.read_csv('train.csv', header=0)
train = train[['VAR_0200', 'VAR_0237', 'VAR_0241']]

test = pd.read_csv('test.csv', header=0)
test = test[['VAR_0200', 'VAR_0237', 'VAR_0241']]


for col in train.columns:
    x = train[col].mode().iloc[0]
    if train[col].isnull().sum() > 0:
        train[col] = train[col].fillna(x)
    if test	[col].isnull().sum() > 0:
        test[col] = test[col].fillna(x)


print(train['VAR_0200'].mode())
print(train['VAR_0200'].isnull().sum())
print(test['VAR_0200'].isnull().sum())

print(train['VAR_0237'].mode())
print(train['VAR_0237'].isnull().sum())
print(test['VAR_0237'].isnull().sum())

print(train['VAR_0241'].mode())
print(train['VAR_0241'].isnull().sum())
print(test['VAR_0241'].isnull().sum())


df = pd.DataFrame(columns=['VAR_TEST','VAR_0200','Percent'])

dict1 = dict()
dict2 = dict()

for index, row in test.iterrows():
	temp = row['VAR_0200'] + row['VAR_0237'] + str(row['VAR_0241'])
	if not temp in dict1:
		choices = train[(train['VAR_0237'].astype(str) == row['VAR_0237']) & (train['VAR_0241'] == row['VAR_0241'])][['VAR_0200']]
		choice_list = choices['VAR_0200'].tolist()
		if not choice_list:
			df = df.append({ 'VAR_TEST':row['VAR_0200'],
						'VAR_0200':row['VAR_0200'],
						'Percent':str("NA")},
						ignore_index = True)
		else:
			try:
				match = process.extractOne(row['VAR_0200'], choice_list)
			except Exception as e:
				print(str(e), row['VAR_0200'])
				print(index, choice_list)
				break
			if (match[1] > 50):
				df = df.append({ 'VAR_TEST':row['VAR_0200'],
							'VAR_0200':str(match[0]),
							'Percent':str(match[1])},
							ignore_index = True)
				dict2[row['VAR_0200']] = str(match[0])
				dict1[temp] = True
			else:
				df = df.append({ 'VAR_TEST':row['VAR_0200'],
						'VAR_0200':row['VAR_0200'],
						'Percent':str("NA")},
						ignore_index = True)
	else:
		df = df.append({'VAR_TEST':row['VAR_0200'],
						'VAR_0200':dict2[row['VAR_0200']],
						'Percent':str("0")},
						ignore_index = True)
	print(index)

df['VAR_TRAIN'] = train['VAR_0200']

df.to_csv("VAR_0200.csv", index = False)
