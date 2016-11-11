import numpy as np # Matrix manipulation
import scipy.io.wavfile as wav   # Reads wav file
import glob # File manipulation
import pandas as pd 
import os
from sklearn import linear_model
from sklearn.cross_validation import train_test_split



IEMOCAP_LOCATION = ".."
LABEL_LOCATION = "./data/"
avgAmps = {}
for session in range(1,6):
    for directory in os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'):
        for filename in glob.glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
            audio = wav.read(filename)
            avgAmp = abs(audio[1]).mean()
            name = filename.split('/')[-1][:-4]
            avgAmps[name] = avgAmp

df = pd.DataFrame.from_dict(avgAmps,orient='index').reset_index()
df.columns = ['session','Average amplitude']
df = df.sort_values(by='session')
df.to_csv('./averageAmplitude.csv',index=False)
