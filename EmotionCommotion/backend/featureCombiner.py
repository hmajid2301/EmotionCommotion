#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:59:59 2016

@author: olly
"""

# In[1]
import pandas as pd
import glob
sessions = pd.read_csv('./data/allLabels.csv')


df = pd.read_csv('./features/averageAmplitude.csv')
for filename in glob.glob('./features/*.csv'):
    new_feature = pd.read_csv(filename)
    df = pd.merge(df,new_feature)

df = pd.merge(sessions,df).drop(['time','label'],axis=1)
df.to_csv('./data/allFeatures.csv',index=False)