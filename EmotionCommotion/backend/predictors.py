from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.models import load_model
import pickle
import sys
sys.path.insert(0, './backend/')
sys.path.insert(0, './app/')
sys.path.insert(0, '../app/')

from datagrabber import *
from allExtractors import *

SCALAR_LOCATION = 'backend/deeplearning/scaler.sav'

cnn_scaler = pickle.load(open(SCALAR_LOCATION,'rb'),encoding='latin1') # encoding for python 2 pickle
cnn = load_model('backend/deeplearning/cnn_15.h5')

def index_to_label(index):
    if index == 0:
        label = 'neu'
    elif index == 1:
        label = 'hap'
    elif index == 2:
        label = 'sad'
    else:
        label = 'ang'
    return label

def label_to_barray(label):
    if label == "neu":
        barray = np.array([1,0,0,0]).reshape(1,4)
    elif label == "hap":
        barray = np.array([0,1,0,0]).reshape(1,4)
    elif label == "sad":
        barray = np.array([0,0,1,0]).reshape(1,4)
    else:
        barray = np.array([0,0,0,1]).reshape(1,4)

    return barray

def index_to_barray(index):
    if index == 0:
        barray = np.array([1,0,0,0]).reshape(1,4)
    elif index == 1:
        barray = np.array([0,1,0,0]).reshape(1,4)
    elif index == 2:
        barray = np.array([0,0,1,0]).reshape(1,4)
    else:
        barray = np.array([0,0,0,1]).reshape(1,4)
    return barray



def svmPredict(audiofile):
    features = [amplitude,energy,f0,silence_ratio,zerocrossing,cepstrum,mfcc]
    frames = get_frames(audiofile)

    agg_vals = []
    for feature in features:
        vals = []
        for frame in frames:
            vals.append(feature(frame, audiofile))
        vals = np.array(vals)
        agg_vals = np.concatenate((agg_vals,aggregate(vals)), axis=0)

    training = pd.read_csv('backend/data/allFeatures.csv')
    training.drop('session',axis=1,inplace=True)
    training = training.replace([np.inf, -np.inf], np.nan)
    training = training.fillna(0)

    training = training.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)


    agg_vals = np.append(agg_vals[0:18],agg_vals[20:])
    agg_vals = agg_vals.reshape(1,-1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(training)
    agg_vals_scaled = min_max_scaler.transform(agg_vals)
    #print(agg_vals_scaled)


    svm = joblib.load('backend/classifiers/svm.pkl')
    result = svm.predict(agg_vals_scaled)
    return result

def cnnPredict(audiofile):
    print("Predicting!")
    # Preprocess the audiofile
    frames = get_frames(audiofile)
    print("Got dem frames")
    scaled = cnn_scaler.transform(frames[0].reshape(1,-1))
    print("Scaled")
    pca = PCA(n_components=40,whiten=True)
    print("pca")
    specto = np.array(signal.spectrogram(scaled,nperseg=128)[2]).reshape(65,142)
    print("specto")
    whitened_specto = pca.fit_transform(specto).reshape(1,65,40,1)
    print("white")
    # Get prediction
    result = cnn.predict(whitened_specto,verbose=0)
    print("result")
    print(result)

    return result
