import numpy as np
import os
from glob import glob

# Read and process data
labels = pd.read_csv('/dcs/project/emotcomm/EmotionCommotion/EmotionCommotion/backend/data/allLabels.csv')
labels = labels.drop('time',axis=1)
labels = labels.set_index('session')
IEMOCAP_LOCATION = '/dcs/project/emotcomm/local/'

clip_labels = np.zeros((40000,4))
j = 0

# Get label for each file in IEMOCAP
for session in range(1,6):
    print('\n' + "Extracting from session: " + str(session) + '\n')
    numdir = len(os.listdir(IEMOCAP_LOCATION + 'IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
    for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
        sys.stdout.write("\r%d%%" % ((i/numdir)*100))
        sys.stdout.flush()
        for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
            name = filename.split('/')[-1][:-4]
            label = labels.loc[name].label
            # Convert from string to binary array
            if label == "neu":
                clip_labels[j] = np.array([1,0,0,0]).reshape(1,4)
            elif label == "hap":
                clip_labels[j] = np.array([0,1,0,0]).reshape(1,4)
            elif label == "sad":
                clip_labels[j] = np.array([0,0,1,0]).reshape(1,4)
            else:
                clip_labels[j] = np.array([0,0,0,1]).reshape(1,4)
            j+=1
        print(j)

# Resize array
clip_labels = clip_labels[0:j]
# Save labels
np.save("clip_labels.npy", clip_labels)
