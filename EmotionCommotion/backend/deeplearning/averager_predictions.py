import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import datasets, linear_model

preds = np.load("preds_10_epoch.npy")
num_frames = np.load("session5_num_frames.npy")#[30242:]
clip_labels = np.load("clip_labels.npy")

# f = num_frames[0]
# print f
# file_preds = preds[start:start + f]
# print file_preds
# print file_preds.mean(axis=0)
# print  np.argmax(file_preds.mean(axis=0))
# j =0
# if pred == 0:
#     y[j] = np.array([1,0,0,0]).reshape(1,4)
# elif pred == 1:
#     y[j] = np.array([0,1,0,0]).reshape(1,4)
# elif pred == 2:
#     y[j] = np.array([0,0,1,0]).reshape(1,4)
# else:
#     y[j] = np.array([0,0,0,1]).reshape(1,4)
# print y[j]



########
# VOTING PREDICTIONS
def mode_predictions(preds, num_frames):
    start = 0
    y = np.zeros((len(num_frames),4))
    j = 0
    for f in num_frames:
        file_preds = preds[start:start + f]
        max_indexes = np.argmax(file_preds,axis=1)
        mode = stats.mode(max_indexes)[0][0]
        if mode == 0:
            y[j] = np.array([1,0,0,0]).reshape(1,4)
        elif mode == 1:
            y[j] = np.array([0,1,0,0]).reshape(1,4)
        elif mode == 2:
            y[j] = np.array([0,0,1,0]).reshape(1,4)
        else:
            y[j] = np.array([0,0,0,1]).reshape(1,4)
        start+=f
        j+=1
    return y
#
y1 = mode_predictions(preds, num_frames)
print "MODE"
print accuracy_score(y1,clip_labels[3548:])


########
#MEAN PREDICTIONS
def mean_predictions(preds, num_frames):
    start = 0
    y = np.zeros((len(num_frames),4))
    j = 0
    for f in num_frames:
        file_preds = preds[start:start + f]
        pred = np.argmax(file_preds.mean(axis=0))
        if pred == 0:
            y[j] = np.array([1,0,0,0]).reshape(1,4)
        elif pred == 1:
            y[j] = np.array([0,1,0,0]).reshape(1,4)
        elif pred == 2:
            y[j] = np.array([0,0,1,0]).reshape(1,4)
        else:
            y[j] = np.array([0,0,0,1]).reshape(1,4)

        start+=f
        j+=1
    return y
########

y2 = mean_predictions(preds, num_frames)
print "MEAN"
print accuracy_score(y2,clip_labels[3548:])


########
#MIXED PREDICTIONS
def mixed_predictions(preds, num_frames):
    start = 0
    y = np.zeros((len(num_frames),4))
    j = 0
    for f in num_frames:
        file_preds = preds[start:start + f]
        mean = file_preds.mean(axis=0)
        base = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        file_preds = np.concatenate((file_preds, base))
        #file_preds.np.([1,0,0,0])
        #file_preds.append([0,1,0,0])
        #file_preds.append([0,0,1,0])
        #file_preds.append([0,0,0,1])

        max_indexes = np.argmax(file_preds,axis=1)
        max_indexes = stats.itemfreq(max_indexes)
        max_indexes = max_indexes[:,1]
        max_indexes = np.apply_along_axis(lambda x: (x-1)*100/float(f),0,max_indexes)
        #print max_indexes
        mean = file_preds.mean(axis=0)
        #print mean

        new = np.add(max_indexes, mean)
        index = np.argmax(new)
        #print new
        #print new
        #print index
        #break

        if index == 0:
            y[j] = np.array([1,0,0,0]).reshape(1,4)
        elif index == 1:
            y[j] = np.array([0,1,0,0]).reshape(1,4)
        elif index == 2:
            y[j] = np.array([0,0,1,0]).reshape(1,4)
        else:
            y[j] = np.array([0,0,0,1]).reshape(1,4)

        start+=f
        j+=1
    return y
########

y3 = mixed_predictions(preds, num_frames)
print "MIXED"
print accuracy_score(y3,clip_labels[3548:])
ind_score =  1/float(len(num_frames))
print (0.559447983015 - 0.547770700637) / ind_score #how many more mode gets right than mean (11)



########
#PRODUCT PREDICTIONS
def prod_predictions(preds, num_frames):
    start = 0
    y = np.zeros((len(num_frames),4))
    j = 0
    for f in num_frames:
        file_preds = preds[start:start + f]
        pred = np.argmax(file_preds.prod(axis=0))
        if pred == 0:
            y[j] = np.array([1,0,0,0]).reshape(1,4)
        elif pred == 1:
            y[j] = np.array([0,1,0,0]).reshape(1,4)
        elif pred == 2:
            y[j] = np.array([0,0,1,0]).reshape(1,4)
        else:
            y[j] = np.array([0,0,0,1]).reshape(1,4)

        start+=f
        j+=1
    return y
########

def mode_of_classifiers(pred_list, num_frames):
  j=0
  y = np.zeros((len(num_frames),4))
  for i in range(len(num_frames)):
    #print pred_list[:,i]
    y[j] = stats.mode(pred_list[:,i])[0][0]
    j+=1
  return y



y4 = prod_predictions(preds, num_frames)
print "PRODUCT"
print accuracy_score(y4,clip_labels[3548:])
print (0.561571125265 - 0.559447983015) / ind_score
comb = np.array((y1,y2,y4))
y_comb = mode_of_classifiers(comb, num_frames)
print "COMBINED"
print accuracy_score(y_comb,clip_labels[3548:])

# print len(y)
#
'''
labels = pd.read_csv('/dcs/project/emotcomm/EmotionCommotion/EmotionCommotion/backend/data/allLabels.csv')
session_5_labels = labels[3548:]
true_labels = np.zeros((40000,4))
j = 0

for label in session_5_labels.label:
    if label == "neu":
        true_labels[j] = np.array([1,0,0,0]).reshape(1,4)
    elif label == "hap":
        true_labels[j] = np.array([0,1,0,0]).reshape(1,4)
    elif label == "sad":
        true_labels[j] = np.array([0,0,1,0]).reshape(1,4)
    else:
        true_labels[j] = np.array([0,0,0,1]).reshape(1,4)
    j+=1

true_labels = true_labels[0:j]
accuracy_score(y,true_labels)




preds_proba = np.load("preds.npy")
y_test = np.load("y_test.npy")
preds = np.zeros((40000,4))
j = 0
for p in preds_proba:
    pred = np.argmax(p)
    if pred == 0:
        preds[j] = np.array([1,0,0,0]).reshape(1,4)
    elif pred == 1:
        preds[j] = np.array([0,1,0,0]).reshape(1,4)
    elif pred == 2:
        preds[j] = np.array([0,0,1,0]).reshape(1,4)
    else:
        preds[j] = np.array([0,0,0,1]).reshape(1,4)

    start+=f
    j+=1
preds = preds[0:j]
accuracy_score(preds,y_test)
'''
