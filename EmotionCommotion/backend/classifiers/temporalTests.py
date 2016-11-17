# -*- coding: utf-8 -*-

# In[1]
import pandas as pd

X = pd.read_pickle('../features/zeroCrossing(1session(s)).pkl')
y = pd.read_csv('../data/allLabels.csv')
y = y.drop('time',axis=1)

# In[2]

X = pd.merge(X,y,how='left').dropna()

# In[3]
X['zeroCrossing'] = X['zeroCrossing'].apply(lambda x: x.flatten())

# In[4]
X['meanZero'] = X['zeroCrossing'].apply(lambda x: x.mean())
X['varZero'] = X['zeroCrossing'].apply(lambda x: x.var())

# In[5]
#X.groupby('label')['meanZero'].mean().plot(kind='bar')
X.groupby('label')['varZero'].mean().plot(kind='bar')