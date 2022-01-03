# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:06:50 2022

@author: parijat
"""

import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
import joblib

pd.pandas.set_option('display.max_columns',None)

data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYekml')

data = data.replace('?',np.nan) 


data.drop(['boat','body','home.dest','name','ticket'], inplace = True, axis = 1)

data.head()

## num and char values

data['age'] = data['age'].astype('float')
data['fare'] = data['fare'].astype('float')


var_char = [var for var in data.columns if data[var].dtypes == 'O' and var != 'survived']
var_num = [var for var in data.columns if var not in (var_char) and var!= 'survived'] 


## split the rows for cabin

def getfirstrow(row):
    
    try:
        return row.split()[0][0]
    except:
        return np.nan
    
data['cabin'] = data['cabin'].apply(getfirstrow)

'''
def firstletter(row):
    
    return row[0]
    
data['cabin'] = data['cabin'].apply(firstletter)
'''

data['cabin'].unique()

data.head(10)


data[var_char].nunique()
data[var_num].nunique()

print('len of variable {}'.format(len(var_char)))
print(var_char)
print('len of variable {}'.format(len(var_num)))
print(var_num)

data['age'].isnull().sum()/len(data)*100
## distribution

data[var_num].hist(bins = 20, figsize=(10,12))
data[var_char].nunique()


data.to_csv('titanic2.csv',index = False)


## divide data in to trainin and testing

X_train,X_test,y_train,y_test = train_test_split(data.drop('survived',axis = 1),data['survived'],test_size = 0.2, random_state = 0)


for var in ['cabin','embarked']:
    X_train[var].fillna('missing',inplace = True)
    X_test[var].fillna('missing',inplace = True) 
data['cabin'].unique()


for var in var_num:
    X_train[var + '_NA'] = np.where(X_train[var].isnull().sum()>0,1,0)
    median_value = data[var].median()
    X_train[var].fillna(median_value,inplace = True)
    X_test[var + '_NA'] = np.where(X_test[var].isnull().sum()>0,1,0)
    X_test[var].fillna(median_value,inplace = True)     

X_train[var_num].head()

for var in var_num:
    print(var , '   ' , X_train[var].dtypes)

data['pclass'].unique()


##############################

# remove rare labels
X_train['cabin'].value_counts()

def frperc(df,var,perc):
    df = df.copy()
    
    tmp = X_train[var].value_counts()/len(df)
    
    return tmp[ tmp > perc].index

for var in var_char:
    fr_ls = frperc(X_train,var,0.05)
    X_train[var] = np.where(X_train[var].isin(fr_ls),X_train[var], 'Rare')
    X_test[var] = np.where(X_test[var].isin(fr_ls),X_test[var], 'Rare')
    
    
X_train['cabin'].value_counts()    

for var in var_char:
    X_train = pd.concat([X_train,pd.get_dummies(X_train[var],prefix = var, drop_first= True )], axis = 1)
    X_test = pd.concat([X_test,pd.get_dummies(X_test[var], prefix = var, drop_first= True )], axis = 1)

X_train.drop(var_char, axis = 1, inplace = True)
X_test.drop(var_char, axis = 1, inplace = True)

test = [var for var in X_test.columns if var not in (X_train.columns)]
test

X_test['embarked_Rare'] = 0

X_test.head()

variables = [ c for c in X_train.columns]


std = StandardScaler()

std.fit(X_train[variables])

X_train = std.transform(X_train[variables])
X_test = std.transform(X_test[variables])

model = LogisticRegression(C = 0.001, random_state = 0)
model.fit(X_train,y_train)

class_ = model.predict(X_train)
pred = model.predict_proba( X_train)[:,1]  # prob of occurance of event i.e. 1 (other wise we get both)
pred

print('ROC AUC {}'.format(roc_auc_score(y_train,pred)))
print('Accuracy Score {}'.format(accuracy_score(y_train,class_)))


class_ = model.predict(X_test)
pred = model.predict_proba(X_test)[:,1]

print('ROC AUC {}'.format(roc_auc_score(y_test,pred)))
print('Accuracy Score {}'.format(accuracy_score(y_test,class_)))











    
        
        
        
        
        