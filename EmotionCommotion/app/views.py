"""
Definition of views.
"""

from django.shortcuts import render, render_to_response
from django.http import HttpRequest, HttpResponse
from django.template import RequestContext
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from sklearn.externals import joblib
import scipy.io.wavfile
import numpy as np
import os, json, sys

import pandas as pd1



from scipy import signal

sys.path.append("./")
sys.path.insert(0, './backend/')
import matplotlib.pyplot as plt


from predictors import *
from .allExtractors import *

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )



@csrf_exempt
def blob(request):
    '''renders blob backend'''


    #From ajax get the information
    #frame - audio file
    #frameNum - number of the frame (since user started recording)
    frame = request.FILES['blob']
    frameNum = request.POST['frame-number']
    print("Frame Num: ", frameNum)
    #if (frameNum == "1"):
    #    print("Resetting summed probs")
    #    summed_path = 'backend/summed_probs.npy'
    #    tot_frames = 'backend/tot_frames.txt'
    #    np.savetxt(summed_path, np.array([[0,0,0,0]]))
    #    open(tot_frames, 'w').write('%d' % 0)

    filename = 'frame' + frameNum + '.wav'
    path = default_storage.save('tmp/' + filename, ContentFile(frame.read()))


    #convert to array of amplitudes
    #convert from stereo to mono
    mydata = scipy.io.wavfile.read(path)
    mydata = mydata[1][:,0]


    #delete file
    if os.path.isfile(path):
        os.remove(path)

    audiofile = get_audiofile(filename,data=mydata,read_file=False,frame_size=16000)

    result = svmPredict(audiofile)
    # Get index of largest probability
    index = np.argmax(result)
    # Get string label from index
    label = index_to_label(index)


    #return predictions
    return HttpResponse(json.dumps({'neu': format(result[0][0], '.2f'),
                                    'hap': format(result[0][1], '.2f'),
                                    'sad': format(result[0][2], '.2f'),
                                    'ang': format(result[0][3], '.2f'),
                                    'frame': frameNum})

                                    , content_type="application/json")
