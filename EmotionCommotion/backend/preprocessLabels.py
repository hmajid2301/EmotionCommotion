import pandas as pd
import glob

IEMOCAP_LOCATION = "../../../../local"
SAVE_LOCATION = "./data/"

for n in range(1,6):
    label_file = pd.DataFrame(columns=['time','session','label'])
    for filename in glob.glob(IEMOCAP_LOCTION + "/IEMOCAP_full_release/Session" + str(n) + "/dialog/EmoEvaluation/*.txt"):
        # Read original label file
        f = pd.read_csv(filename,sep=('\t'),names=['time','session','label'])
        # Ignore first row
        f = f[1:]
        f = f[f.session.str.contains('^Ses')]
        label_file = label_file.append(f)
        label_file = label_file.sort_values(by='session')
        label_file.to_csv(SAVE_LOCATION + 'session' + str(n) + 'Labels.csv',index=False)
