from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.models import load_model
import pickle
import sys
import pandas as pd
sys.path.insert(0, '../../../backend/')
sys.path.insert(0, '../../../app/')



import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
global graph

SCALAR_LOCATION = '../../../backend/deeplearning/scaler.sav'

cnn_scaler = pickle.load(open(SCALAR_LOCATION,'rb'),encoding='latin1') # encoding for python 2 pickle
cnn = load_model('../cnns/sessions123_cnn.h5')
graph = tf.get_default_graph()

IEMOCAP_LOCATION = '../../../data'
#For every file
columns = 'neu_vote,hap_vote,sad_vote,ang_vote,neu_2nd,hap_2nd,sad_2nd,ang_2nd,neu_3rd,hap_3rd,sad_3rd,ang_3rd,neu_min,hap_min,sad_min,ang_min,neu_mean,hap_mean,sad_mean,ang_mean,neu_var,hap_var,sad_var,ang_var,neu_q1,hap_q1,sad_q1,ang_q1,neu_q2,hap_q2,sad_q2,ang_q2,neu_max,hap_max,sad_max,ang_max,neu_min,hap_min,sad_min,ang_min'

from datagrabber import *
def aggregate_cnn_IEMOCAP(verbose=2,aggregate=True):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the IMEOCAP database, and saves the results
    in the feaures directory.
    '''
    # Fill a dict with values
    dic = {}
    vals = []
    filenames = []
    for session in range(4,6):
        if verbose > 0:
            print('\n' + "Extracting from session: " + str(session) + '\n')
            numdir = len(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
        for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
            if verbose > 1:
                sys.stdout.write("\r%d%%" % ((i/numdir)*100))
                sys.stdout.flush()
            for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
                name = filename.split('/')[-1][:-4]
                filenames.append(name)
                audiofile = get_audiofile(filename, frame_size=16000)
                frames = np.array(get_frames(audiofile))
                #Use cnn to make prediction for each frame
                frame_predictions = np.apply_along_axis(cnn_frame_predict, 1, frames)
                frame_agg_predictions = aggregate_frame_preditions(frame_predictions)
                #Save with label?
                vals.append(frame_agg_predictions)
    dtype = [('Filename','object'), ('Col1','float32'), ('Col2','float32'), ('Col3','float32'), ('Col4','float32'), ('Col5','float32'), ('Col6','float32'), ('Col7','float32'), ('Col8','float32'), ('Col9','float32'), ('Col10','float32'), ('Col11','float32'), ('Col12','float32'), ('Col13','float32'), ('Col14','float32'), ('Col15','float32'), ('Col16','float32'), ('Col17','float32'), ('Col18','float32'), ('Col19','float32'), ('Col20','float32'), ('Col21','float32'), ('Col22','float32'), ('Col23','float32'), ('Col24','float32'), ('Col25','float32'), ('Col26','float32'), ('Col27','float32'), ('Col28','float32'), ('Col29','float32'), ('Col30','float32'), ('Col31','float32'), ('Col32','float32'), ('Col33','float32'), ('Col34','float32'), ('Col35','float32'), ('Col36','float32'), ('Col37','float32'), ('Col38','float32'), ('Col39','float32'), ('Col40','float32')]
    # values = numpy.zeros(20, dtype=dtype)
    df = pd.DataFrame(vals, index=filenames)
    df.to_csv(path_or_buf='IEMOCAP_frame_agg_45.csv', sep=',')
    #np.savetxt("IEMOCAP_frame_agg.csv", np.asarray(vals), delimiter=",", header = columns)

def aggregate_cnn_wild(verbose=2,aggregate=True):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the IMEOCAP database, and saves the results
    in the feaures directory.
    '''
    vals = []
    files = glob('../../../../../local/wild_dataset/10_to_20_seconds/*.wav')
    num_files = len(files)
    i = 0
    filenames = []
    for filename in files:
        name = filename.split('/')[-1][:-4]
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
        filenames.append(name)
    dtype = [('Filename','object'), ('Col1','float32'), ('Col2','float32'), ('Col3','float32'), ('Col4','float32'), ('Col5','float32'), ('Col6','float32'), ('Col7','float32'), ('Col8','float32'), ('Col9','float32'), ('Col10','float32'), ('Col11','float32'), ('Col12','float32'), ('Col13','float32'), ('Col14','float32'), ('Col15','float32'), ('Col16','float32'), ('Col17','float32'), ('Col18','float32'), ('Col19','float32'), ('Col20','float32'), ('Col21','float32'), ('Col22','float32'), ('Col23','float32'), ('Col24','float32'), ('Col25','float32'), ('Col26','float32'), ('Col27','float32'), ('Col28','float32'), ('Col29','float32'), ('Col30','float32'), ('Col31','float32'), ('Col32','float32'), ('Col33','float32'), ('Col34','float32'), ('Col35','float32'), ('Col36','float32'), ('Col37','float32'), ('Col38','float32'), ('Col39','float32'), ('Col40','float32')]
    # values = numpy.zeros(20, dtype=dtype)
    df = pd.DataFrame(vals, index=filenames)

    #format="%s,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e"
    #np.savetxt("wild_frame_agg.csv", np.array(vals), delimiter=",", header = columns)
    df.to_csv(path_or_buf='wild_frame_agg.csv', sep=',')

def aggregate_frame_preditions(frame_predictions):
    #Calculate sum for each emotion
    positions =  np.argsort(frame_predictions)
    modal_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    sum_max = np.sum(modal_matrix[positions[:,0]], axis=0)
    sum_2nd = np.sum(modal_matrix[positions[:,1]], axis=0)
    sum_3rd = np.sum(modal_matrix[positions[:,2]], axis=0)
    sum_min = np.sum(modal_matrix[positions[:,3]], axis=0)
    #Calculate mean for each emotion
    frame_means = np.mean(frame_predictions, axis = 0)
    #Calculate variance for each emotion
    frame_vars = np.var(frame_predictions, axis = 0)
    frame_q1 = np.percentile(frame_predictions, 25, axis=0)
    frame_q2 = np.percentile(frame_predictions, 75, axis=0)
    frame_max = np.max(frame_predictions, axis=0)
    frame_min = np.max(frame_predictions, axis=0)
    #Concatenate results
    frame_agg_predictions = np.concatenate([sum_max, sum_2nd, sum_3rd, sum_min,frame_means, frame_vars, frame_q1, frame_q2, frame_max, frame_min])
    return frame_agg_predictions

def cnn_frame_predict(frame):
    scaled = cnn_scaler.transform(frame.reshape(1,-1))
    pca = PCA(n_components=40,whiten=True)
    specto = np.array(signal.spectrogram(scaled,nperseg=128)[2]).reshape(65,142)
    whitened_specto = pca.fit_transform(specto).reshape(1,65,40,1)
    # Fixed threading problem: https://github.com/fchollet/keras/issues/2397
    with graph.as_default():
        # Get prediction
        result = cnn.predict(whitened_specto,verbose=0)
        result = result.reshape(4)
        return result

#aggregate_cnn_wild()
aggregate_cnn_IEMOCAP()
