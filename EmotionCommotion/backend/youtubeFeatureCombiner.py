import pandas as pd
import glob

# Load session labels
sessions = pd.read_csv('./data/emmotion_labels.csv')

# Load first feature
df = pd.read_csv('./features/youtube_standardized/youtube_amplitude_standardized.csv')

# Add other features
for filename in glob.glob('./features/youtube_standardized/*.csv'):
    new_feature = pd.read_csv(filename)
    df = pd.merge(df,new_feature)

# Merge features with lables to ensure all features and lables and paired,
# then drop label.
df = pd.merge(sessions,df).drop(['emotion'],axis=1)
df = df.sort_index(axis=1)

# Save csv
df.to_csv('./data/allYoutubeFeaturesStandardized.csv',index=False)
