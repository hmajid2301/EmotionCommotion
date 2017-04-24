#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:59:59 2016
@author: olly
"""
import pandas as pd
import glob

# Load session labels
sessions = pd.read_csv('./data/allLabels.csv')

# Load first feature
df = pd.read_csv('./features/iemocap_standardized/amplitude_standardized.csv')

# Add other features
for filename in glob.glob('./features/iemocap_standardized/*.csv'):
    new_feature = pd.read_csv(filename)
    df = pd.merge(df,new_feature)

# Merge features with lables to ensure all features and lables and paired,
# then drop label.
df = pd.merge(sessions,df).drop(['time','label'],axis=1)
df = df.sort_index(axis=1)

# Save csv
df.to_csv('./data/allFeatures_standardized.csv',index=False)
