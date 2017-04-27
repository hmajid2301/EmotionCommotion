from sklearn import preprocessing
from sklearn.decomposition import PCA
#from keras.models import load_model
import pickle
import sys

from sklearn.externals import joblib
sys.path.insert(0, './backend/')
sys.path.insert(0, './app/')
sys.path.insert(0, '../app/')

from datagrabber import *
from allExtractors import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#import tensorflow as tf
global graph

SCALAR_LOCATION = 'backend/deeplearning/scaler.sav'
FRAME_SCALER_LOCATION  = "backend/frame_scaler.sav"

#cnn_scaler = pickle.load(open(SCALAR_LOCATION,'rb'),encoding='latin1') # encoding for python 2 pickle
#cnn = load_model('backend/classifiers/cnns/all_data_cnn.h5')
#graph = tf.get_default_graph()

# Load pre-trained scaler
scaler = pickle.load(open(SCALER_LOCATION, 'rb'),encoding='latin1')
frame_scaler = pickle.load(open(FRAME_SCALER_LOCATION, 'rb'),encoding='latin1')

def index_to_label(index):
    '''
    Converts an interger index to a label string
    '''
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
    '''
    Converts a label string to a binary array
    '''
    if label == "neu" or label=="Neutral":
        barray = np.array([1,0,0,0]).reshape(1,4)
    elif label == "hap" or label=="Happy":
        barray = np.array([0,1,0,0]).reshape(1,4)
    elif label == "sad" or label== "Sad":
        barray = np.array([0,0,1,0]).reshape(1,4)
    else:
        barray = np.array([0,0,0,1]).reshape(1,4)

    return barray

def index_to_barray(index):
    '''
    Converts an integer index to a binary array
    '''
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
    '''
    Use a pre-trained SVM to predict emotion from audiofile
    '''
    # List of features to be extracted
    features = [amplitude,zerocrossing,cepstrum,mfcc,f0,energy,silence_ratio]

    # Split audio into frames
    frames = get_frames(audiofile)

    agg_vals = []
    for feature in features:
        vals = []
        for frame in frames:
            frame = frame_scaler.transform(frame.reshape(1,-1))
            frame = frame.reshape((16000,))

            # Extract feature from frame
            vals.append(feature(frame, audiofile))

        vals = np.array(vals)
        #print(feature.__name__ ,vals.shape)

        # Aggregate values
        feature_agg_vals = aggregate(vals)
        agg_vals = np.concatenate((agg_vals,feature_agg_vals), axis=0)
        #print(feature_agg_vals)

    agg_func_names = ["max", "mean", "var"]

    # for i in range(0, len(agg_func_names)):
    #     for j in range(0, len(labels)):
    #         print((agg_func_names[i]+'('+labels[j]+'('+features.__name__+'))'))

    # Load data to get scaler
    training = pd.read_csv('backend/data/allFeatures_standardized.csv')
    training.drop('session',axis=1,inplace=True)
    training = training.replace([np.inf, -np.inf], np.nan)
    training = training.fillna(0)
    #print(training.columns.values)
    training = training.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)

    # Remove max and mean zerocrossing features
    #print(agg_vals.shape)
    agg_vals = np.append(agg_vals[0:9],agg_vals[11:])
    #print(agg_vals.shape)
    agg_vals = agg_vals.reshape(1,-1)
    #print(agg_vals)
    min_max_scaler = preprocessing.MinMaxScaler()

    # Fit scaler to traning data
    min_max_scaler.fit(training)

    # Scale values
    agg_vals_scaled = min_max_scaler.transform(agg_vals)
    # Load SVM and use to predict emotion
    svm = joblib.load('backend/classifiers/saved_classifiers/svm.pkl')
    #svm = pickle.load(open('backend/classifiers/saved_classifiers/svm.pkl', 'rb'))
    result = svm.predict_proba(agg_vals_scaled)


    summed_path = 'backend/summed_probs.npy'
    tot_frames_path = 'backend/tot_frames.txt'


    #if os.path.isfile(path):
    #os.remove(path)
    summed_probs = np.loadtxt(summed_path).reshape(1,4)
    summed_probs = summed_probs + result

    f = open(tot_frames_path, 'r')
    for line in f.readlines():
        tot_frames = int(line) + 1

    print("tot_frames",tot_frames)
    print(result)
    print(summed_probs/tot_frames)

    np.savetxt(summed_path, summed_probs)
    f = open(tot_frames_path, 'w')
    f.write('%d' % tot_frames)


    return summed_probs/tot_frames

def cnnPredict(audiofile):
    '''
    Use a pre-trained CNN to predict emotion from audiofile
    '''
    # Split audio into frames
    frames = get_frames(audiofile)

    # Scale frames
    scaled = cnn_scaler.transform(frames[0].reshape(1,-1))
    pca = PCA(n_components=40,whiten=True)
    # Generate spectrogram
    specto = np.array(signal.spectrogram(scaled,nperseg=128)[2]).reshape(65,142)
    # PCA whitening
    whitened_specto = pca.fit_transform(specto).reshape(1,65,40,1)
    # Fixed threading problem: https://github.com/fchollet/keras/issues/2397
    with graph.as_default():
        # Get prediction
        result = cnn.predict(whitened_specto,verbose=0)
        print(result)

        return result
