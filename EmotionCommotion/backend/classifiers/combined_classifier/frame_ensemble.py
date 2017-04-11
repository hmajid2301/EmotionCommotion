from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import argparse
from classifiers import * 

# Mode constants
SESSION_5 = 0
WILD = 1
STACKER = 2 
# Model constants
SVM = 0
RANDOM_FOREST = 1
LOGISTIC_REGRESSION = 2

# Returns training and test data from the IEMOCAP dataset
# Training data is from the sessions 1-4
# Test data is from session 5
def train_test_IEMOCAP():
    # Read in the aggregated features provided by the cnn on the IEMOCAP database
    X = pd.read_csv('./IEMOCAP_frame_agg.csv')
    # Read in the labels for each file in the IEMOCAP database
    y = pd.read_csv('../../data/allLabels.csv')
    #Ensure they match up together
    X.sort_values(by='session')
    y.sort_values(by='session')
    filenames = X['session']
    #Drop unnecessary columns
    X = np.ravel(X.drop(['session'],axis=1)).reshape((-1,40))
    y = np.ravel(y.drop(['session','time'],axis=1))
    #Session 5 for testing
    #3548 is the first sample from session 5
    X_train = X[:3548]
    X_test = X[3548:]
    y_train = y[:3548]
    y_test = y[3548:]
    return [X_train, X_test, y_train, y_test, filenames]

# Training data is from the IEMOCAP dataset
# Test data is from the wild dataset
def train_test_wild():
    # Read in the aggregated features provided by the cnn on the IEMOCAP database
    X_train = pd.read_csv('./IEMOCAP_frame_agg.csv')
    # Read in the labels for each file in the IEMOCAP database
    y_train = pd.read_csv('../../data/allLabels.csv')
    # Read in the aggregated features provided by the cnn on the wild database
    X_test = pd.read_csv('./wild_frame_agg.csv')
    # Read in the labels for each file in the wild database
    y_test = pd.read_csv('./wild_dataset/wild_dataset/emotion_labels.csv')
    filenames = X_train['session']
    #Ensure they match up together
    X_train.sort_values(by='session')
    y_train.sort_values(by='session')
    X_test.sort_values(by='session')
    y_test.sort_values(by='session')
    # Drop unnecessary columns
    X_train = np.ravel(X_train.drop(['session'],axis=1)).reshape((-1,40))
    y_train = np.ravel(y_train.drop(['session','time'],axis=1))
    X_test = np.ravel(X_test.drop(['session'],axis=1)).reshape((-1,40))
    y_test = np.ravel(y_test.drop(['session'],axis=1))
    return [X_train, X_test, y_train, y_test, filenames]

# Split the IEMOCAP training sets in two
def stacker_split_train(X_train, y_train):
    X_train_a = X_train[:1755]
    X_train_b = X_train[1755:]
    y_train_a = y_train[:1755]
    y_train_b = y_train[1755:]
    return [X_train_a, X_train_b, y_train_a, y_train_b]

# Generate and save probability predictions for use in a stacker model
def stacker(X_train, y_train, X_test, filenames):
    [X_train_a, X_train_b, y_train_a, y_train_b] = stacker_split_train(X_train, y_train)
    model_a = get_model(SVM, X_train_a, y_train_a)
    model_b = get_model(SVM, X_train_b, y_train_b)
    model_both = get_model(SVM, X_train, y_train)
    pred_prob_a = model_a.predict_proba(X_train_b)
    pred_prob_b = model_a.predict_proba(X_train_a)
    pred_prob_both = model_both.predict_proba(X_test)
    pred_proba = np.concatenate([pred_prob_a, pred_prob_b, pred_prob_both])
    df = pd.DataFrame(pred_proba, index=filenames)
    df.to_csv(path_or_buf='svm_proba2.csv', sep=',')
    return model_both

# Return a model fitted with the training data provided
def get_model(model_type, X_train, y_train):
    if model_type == SVM:
        model = svm_combined_predictor(X_train, y_train)
    elif  model_type == RANDOM_FOREST:
        model = rforest_combined_predictor(X_train, y_train)
    elif model_type == LOGISTIC_REGRESSION:
        model = logistic_combined_predictor(X_train, y_train)
    return model

# Determine model type to use based on input args
def get_model_type(model_arg):
    model_arg = model_arg.lower()
    #Default
    mode = SVM
    if model_arg == "svm" or model_arg == "s":
        model = SVM
    elif model_arg == "random_forest" or model_arg == "rf":
        model = RANDOM_FOREST
    elif model_arg == "logistic_regression" or model_arg == "lr":
        model = LOGISTIC_REGRESSION
    return model

# Determine mode to use based on input args
def get_mode_type(mode_arg):
    model_arg = mode_arg.lower()
    #Default
    mode = SESSION_5
    if mode_arg == "s5":
        mode = SESSION_5
    elif mode_arg == "w":
        mode = WILD
    elif mode_arg == "st":
        mode = STACKER
    return mode

def main(args):
    # Determine model and mode based on user inputs
    mode = get_mode_type(args.mode)
    model_type = get_model_type(args.model)
    cm_filename = args.cm
    if mode == SESSION_5:
        [X_train, X_test, y_train, y_test, filenames] = train_test_IEMOCAP()
        [X_train, X_test] = preprossess_attributes(X_train, X_test)
        model = get_model(model_type, X_train, y_train)
    elif mode == WILD:
        [X_train, X_test, y_train, y_test, filenames] = train_test_wild()
        [X_train, X_test] = preprossess_attributes(X_train, X_test)
        model = get_model(model_type, X_train, y_train)
    elif mode == STACKER:
        [X_train, X_test, y_train, y_test, filenames] = train_test_IEMOCAP()
        [X_train, X_test] = preprossess_attributes(X_train, X_test)
        model = stacker(X_train, y_train, X_test, filenames)
        
    preds = model.predict(X_test)
    calculate_accuracy(y_test, preds, plot=True, save_as=cm_filename)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("model",help="'svm','s' -> Support Vector Machine, 'random_forest','rf' -> Random Forest, 'logistic_regression', 'rf' -> Logistic Regression")
    parser.add_argument("mode",help="'s5' -> Test on IEMOCAP Session 5, 'w' -> Test on wild dataset, 'st' -> Save stacker training csv")
    parser.add_argument("cm",help="Filename of confusion matrix to be saved")
    

    args = parser.parse_args()
    main(args)
