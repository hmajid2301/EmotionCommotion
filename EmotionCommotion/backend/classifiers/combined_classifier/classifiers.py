from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model, datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np


def preprossess_attributes(X_train, X_test):
    #Scale data
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    #PCA
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return [X_train, X_test]

# Scoring metric
def meanAcc(estimator, X, y):
    preds = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, preds)
    cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    return cm_normalized.trace()/4

def svm_combined_predictor(X_train, y_train):
    print("SVM Model")
    param_grid = [
      {'C': [1], 'kernel': ['linear']}
     ]

    # tuned_parameters = [{'C': [50,100,200,500,1000]}]
    # tuned_parameters = [{'C': [53,103,203]}]
    # tuned_parameters = [{'C': [203]}]

    # param_grid = [
    #   {'C': [1, 10, 100,], 'kernel': ['linear']},
    #   {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    #  ]
    svm = GridSearchCV(SVC(class_weight='balanced', probability=True), param_grid, cv=5,verbose=5,scoring=meanAcc, n_jobs = -1)
    svm.fit(X_train, y_train)
    print("Best parameters: ", svm.best_params_)
    return svm

def logistic_combined_predictor(X_train, y_train):
    print("Logistic Regression Model")
    logreg = linear_model.LogisticRegression()
    param_grid = {'penalty': ['l1','l2'],
                'C': [0.25,0.5,0.75,1.0,1.5,2.0]
                  }
    # this may take some minutes
    logreg = GridSearchCV(logreg, param_grid, n_jobs=-1,verbose=10,scoring=meanAcc)
    logreg.fit(X_train, y_train)
    print("Best parameters: ", logreg.best_params_)
    return logreg

def rforest_combined_predictor(X_train, y_train):
    print("Random Forest Model")
    est = RandomForestClassifier(n_estimators=2000, max_features = 'auto', max_depth=4, min_samples_leaf=3) #2000
    # for i in range(1,10):
    #   print(clf.predict(X[i-1:i]))
    # param_grid = {'max_features': ['auto'],
    #               'max_depth': [4, 6],
    #               'min_samples_leaf': [3, 5],  ## depends on the nr of training examples
    #               'max_features': [1.0, 0.3, 0.1] ## not possible in our example (only 1 fx)
    #               }
    param_grid = {'max_features': ['auto'],
                  'max_depth': [4],
                  'min_samples_leaf': [3],  ## depends on the nr of training examples
                  'max_features': [1.0] ## not possible in our example (only 1 fx)
                  }
    #this may take some minutes
    gs_cv = GridSearchCV(est, param_grid, n_jobs=-1,verbose=10,scoring=meanAcc).fit(X_train, y_train)

    #best hyperparameter setting
    print('Best hyperparameters: %r' % gs_cv.best_params_)      
    gs_cv.fit(X_train, y_train)
    return gs_cv



import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, save_as, title='RF Confusion matrix', cmap=plt.cm.Greens,fontsize=14):
    plt.figure(figsize=(4.5,4.5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=fontsize+3)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize) 

    tick_marks = np.arange(len(['Angry','Happy','Neutral','Sad']))
    plt.xticks(tick_marks, ['Angry','Happy','Neutral','Sad'], rotation=45,fontsize=fontsize)
    plt.yticks(tick_marks, ['Angry','Happy','Neutral','Sad'],fontsize=fontsize)
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]) + "%",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                    fontsize=fontsize)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=fontsize)
    plt.xlabel('Predicted label',fontsize=fontsize)
    plt.gcf().subplots_adjust(bottom=0.25,left=0.25)
    plt.savefig(save_as,transparent=True,figsize=(20,20),dpi=120)

def calculate_accuracy(y_test, preds, plot=False,save_as="wild_svm.png"):
    cnf_matrix = confusion_matrix(y_test, preds)
    cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm_normalized.round(3) * 100
    print("Accuracy: ", cm_normalized.trace()/4)
    plot_confusion_matrix(cm_normalized, save_as)