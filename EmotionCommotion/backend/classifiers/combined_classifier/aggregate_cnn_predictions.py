from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.models import load_model
import pickle
import sys
sys.path.insert(0, '../../../backend/')
sys.path.insert(0, '../../../app/')

from datagrabber import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
global graph

SCALAR_LOCATION = '../../../backend/deeplearning/scaler.sav'

cnn_scaler = pickle.load(open(SCALAR_LOCATION,'rb'),encoding='latin1') # encoding for python 2 pickle
cnn = load_model('../../../backend/deeplearning/cnn_1_60.h5')
graph = tf.get_default_graph()

IEMOCAP_LOCATION = '../../../data'
#For every file

def aggregate_cnn_IEMOCAP(IEMOCAP_LOCATION,verbose=2,aggregate=True):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the IMEOCAP database, and saves the results
    in the feaures directory.
    '''
    columns = 'neu_vote,hap_vote,sad_vote,ang_vote,neu_mean,hap_mean,sad_mean,ang_mean,neu_var,hap_var,sad_var,ang_var,neu_q1,hap_q1,sad_q1,ang_q1,neu_q2,hap_q2,sad_q2,ang_q2,neu_max,hap_max,sad_max,ang_max,neu_min,hap_min,sad_min,ang_min'
    # Fill a dict with values
    dic = {}
    vals = []
    for session in range(1,6):
        if verbose > 0:
            print('\n' + "Extracting from session: " + str(session) + '\n')
            numdir = len(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
        for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
            if verbose > 1:
                sys.stdout.write("\r%d%%" % ((i/numdir)*100))
                sys.stdout.flush()
            for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
                name = filename.split('/')[-1][:-4]
                audiofile = get_audiofile(filename, frame_size=16000)
                frames = np.array(get_frames(audiofile))
                #Use cnn to make prediction for each frame
                frame_predictions = np.apply_along_axis(cnn_frame_predict, 1, frames)
                frame_agg_predictions = aggregate_frame_preditions(frame_predictions)
                #Save with label?
                vals.append(frame_agg_predictions)
    np.savetxt("IEMOCAP_frame_agg.csv", np.asarray(vals), delimiter=",", header = columns)

def aggregate_cnn_wild(verbose=2,aggregate=True):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the IMEOCAP database, and saves the results
    in the feaures directory.
    '''
    columns = 'neu_vote,hap_vote,sad_vote,ang_vote,neu_mean,hap_mean,sad_mean,ang_mean,neu_var,hap_var,sad_var,ang_var,neu_q1,hap_q1,sad_q1,ang_q1,neu_q2,hap_q2,sad_q2,ang_q2,neu_max,hap_max,sad_max,ang_max,neu_min,hap_min,sad_min,ang_min'
    # Fill a dict with values
    dic = {}
    vals = []
    files = glob(IEMOCAP_LOCATION + '/*.wav')
    num_files = len(files)
    i = 0
    for filename in files:
        i = i + 1
        sys.stdout.write("\r%d%%" % ((i/num_files)*100))
        sys.stdout.flush()
        name = filename.split('/')[-1][:-4]
        audiofile = get_audiofile(filename, frame_size=16000)
        frames = np.array(get_frames(audiofile))
        #Use cnn to make prediction for each frame
        frame_predictions = np.apply_along_axis(cnn_frame_predict, 1, frames)
        #Calculate sum for each emotion
        frame_agg_predictions = aggregate_frame_preditions(frame_predictions)
        #Save with label?
        vals.append(frame_agg_predictions)
    np.savetxt("wild_frame_agg.csv", np.asarray(vals), delimiter=",", header = columns)

def aggregate_frame_preditions(frame_predictions):
    #Calculate sum for each emotion
    max_index = np.argmax(frame_predictions, axis = 1)
    modal_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    #labels = [4,3,2,1]
    frame_sums = np.sum(modal_matrix[max_index], axis=0)
    #Calculate mean for each emotion
    frame_means = np.mean(frame_predictions, axis = 0)
    #Calculate variance for each emotion
    frame_vars = np.var(frame_predictions, axis = 0)
    frame_q1 = np.percentile(frame_predictions, 25, axis=0)
    frame_q2 = np.percentile(frame_predictions, 75, axis=0)
    frame_max = np.max(frame_predictions, axis=0)
    frame_min = np.max(frame_predictions, axis=0)
    #Concatenate results
    frame_agg_predictions = np.concatenate([frame_sums,frame_means, frame_vars, frame_q1, frame_q2, frame_max, frame_min])

def cnn_frame_predict(frame):
    scaled = cnn_scaler.transform(frame.reshape(1,-1))
    pca = PCA(n_components=60,whiten=True)
    specto = np.array(signal.spectrogram(scaled,nperseg=128)[2]).reshape(65,142)
    whitened_specto = pca.fit_transform(specto).reshape(1,65,60,1)
    # Fixed threading problem: https://github.com/fchollet/keras/issues/2397
    with graph.as_default():
        # Get prediction
        result = cnn.predict(whitened_specto,verbose=0)
        #print(result)

        return result

aggregate_cnn_wild()



###IGNORE FROM HERE###
###AWFUL WEIRD THINGS LIE AHEAD WHICH I MAY NEED LATER BUT NEED NOT CONCERN YOU###
def frame_time_results(IEMOCAP_LOCATION,verbose=2,aggregate=True):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the IMEOCAP database, and saves the results
    in the feaures directory.
    '''

    # Fill a dict with values
    dic = {}
    vals = []
    for session in range(1,6):
        if verbose > 0:
            print('\n' + "Extracting from session: " + str(session) + '\n')
            numdir = len(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
        for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
            if verbose > 1:
                sys.stdout.write("\r%d%%" % ((i/numdir)*100))
                sys.stdout.flush()
            for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
                name = filename.split('/')[-1][:-4]
                audiofile = get_audiofile(filename, frame_size=16000)
                frames = np.array(get_frames(audiofile))
                #Use cnn to make prediction for each frame
                frame_predictions = np.apply_along_axis(cnn_frame_predict, 1, frames)
                num_frames = len(frames)
                if num_frames > 25:
                    frame_pred_padded = np.resize(frame_predictions,[25,4])
                else:
                    frame_pred_padded = np.pad(frame_predictions,((0,25-num_frames),(0,0)), 'constant', constant_values=0.0)
                vals.append(frame_pred_padded.reshape(-1))
    np.savetxt("frame_pred_time.csv", np.asarray(vals), delimiter=",")

#frame_time_results(IEMOCAP_LOCATION)

stretch_matrix = np.array([[24],[12,12],[8,8,8],[6,6,6,6],[4,5,5,5,5],
                        [4,4,4,4,4,4],[3,3,4,4,4,3,3],[3,3,3,3,3,3,3,3], #8
                        [2,2,3,3,3,3,3,3,2],[2,2,2,3,3,3,3,2,2,2], #9,10
                        [2,2,2,2,2,3,3,2,2,2,2],[2,2,2,2,2,2,2,2,2,2,2,2], #11,12
                        [1,2,2,2,2,2,2,2,2,2,2,2,1],[1,1,2,2,2,2,2,2,2,2,2,2,1,1], #13,14
                        [1,1,1,2,2,2,2,2,2,2,2,2,1,1,1],[1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1], #15,16
                        [1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1], #17,18
                        [1,1,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1], #19,20
                        [1,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1,1],#21,22
                        [1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
                        )
def frame_stretch_results(IEMOCAP_LOCATION,verbose=2,aggregate=True):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the IMEOCAP database, and saves the results
    in the feaures directory.
    '''

    # Fill a dict with values
    dic = {}
    vals = []
    for session in range(1,6):
        if verbose > 0:
            print('\n' + "Extracting from session: " + str(session) + '\n')
            numdir = len(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
        for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
            if verbose > 1:
                sys.stdout.write("\r%d%%" % ((i/numdir)*100))
                sys.stdout.flush()
            for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
                name = filename.split('/')[-1][:-4]
                audiofile = get_audiofile(filename, frame_size=16000)
                frames = np.array(get_frames(audiofile))
                #Use cnn to make prediction for each frame
                frame_predictions = np.apply_along_axis(cnn_frame_predict, 1, frames)
                num_frames = len(frames)
                if num_frames > 24:
                    stretched = np.resize(frame_predictions,[24,4])
                else:
                    stretched = np.repeat(frame_predictions,stretch_matrix[len(frame_predictions)-1], axis=0)
                vals.append(stretched.reshape(-1))
    np.savetxt("frame_pred_time_stretch.csv", np.asarray(vals), delimiter=",")

#frame_stretch_results(IEMOCAP_LOCATION)