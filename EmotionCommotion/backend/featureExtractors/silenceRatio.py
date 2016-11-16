#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:18:26 2016

@author: Olly Styles
"""
# In[1]
import scipy.io.wavfile as wav   # Reads wav file
import pandas as pd
import numpy as np
import os
import glob


# In[2]
IEMOCAP_LOCATION = "../../../../local"
silenceRatio = {}
for session in range(1,6):
    for directory in os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'):
        for filename in glob.glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
            audio = wav.read(filename)[1]
            threshold = max(audio) * 0.03
            thresholdedAudio = audio[np.where(abs(audio) > threshold)]
            ratio = 1 - (len(thresholdedAudio) / len(audio))            
            name = filename.split('/')[-1][:-4]
            silenceRatio[name] = ratio

df = pd.DataFrame.from_dict(silenceRatio,orient='index').reset_index()
df.columns = ['session','Silence ratio']
df = df.sort_values(by='session')
df.to_csv('../features/silenceRatio.csv',index=False)

