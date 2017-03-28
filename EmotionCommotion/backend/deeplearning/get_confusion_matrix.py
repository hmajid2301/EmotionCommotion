from sklearn.decomposition import PCA
import pickle
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cnn = load_model('cnn_quick.h5')

X_test = np.load('../../../../local/whitened_data/X_test_whitened.npy')
y_test = np.load('../../../../local/whitened_data/y_test.npy')

soft_preds = cnn.predict(X_test.reshape(7839,65,40,1))

def plot_confusion_matrix(cm, title='Confusion matrix - Naive Bayes', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(['Angry','Happy','Neutral','Sad']))
    plt.xticks(tick_marks, ['Angry','Happy','Neutral','Sad'], rotation=45)
    plt.yticks(tick_marks, ['Angry','Happy','Neutral','Sad'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.savefig("../results/naiveBayesCM.png")


hard_preds = np.zeros(soft_preds.shape[0])
hard_y = np.argmax(y_test, axis=1)
i = 0
for pred in preds:
    hard_preds[i] = np.argmax(pred)

    # if label == 0:
    #     hard_preds[i] = np.array([1,0,0,0]).reshape(1,4)
    # elif label == 1:
    #     hard_preds[i] = np.array([0,1,0,0]).reshape(1,4)
    # elif label == 2:
    #     hard_preds[i] = np.array([0,0,1,0]).reshape(1,4)
    # else:
    #     hard_preds[i] = np.array([0,0,0,1]).reshape(1,4)
    i = i + 1

cnf_matrix = confusion_matrix(hard_y, hard_preds)
