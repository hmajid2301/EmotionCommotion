
# coding: utf-8

# In[43]:

import pandas as pd
import glob

IEMOCAP_LOCTION = "."
SAVE_LOCATION = "./data/"

for n in range(1,6):
    file = pd.DataFrame(columns=['time','session','label'])
    for filename in glob.glob(IEMOCAP_LOCTION + "/IEMOCAP_full_release/Session" + str(n) + "/dialog/EmoEvaluation/*.txt"):
        f = pd.read_csv(filename,sep=('\t'),names=['time','session','label'])
        f = f[1:]
        f = f[f.session.str.contains('^Ses')]
        file = file.append(f)
        file = file.sort_values(by='session')
        file.to_csv(SAVE_LOCATION + 'session' + str(n) + '.csv',index=False)

