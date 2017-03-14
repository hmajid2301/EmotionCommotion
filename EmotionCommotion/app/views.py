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

    #lastblob = request.FILES['last-blob']
    # RATE = 32000
    # CHUNK = 1024
    # RECORD_SECONDS = 1
    frame = request.FILES['frame']
    # print(type(frame))
    # data2 = []
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(CHUNK)
    #     data2.append(data)
    # joined = ''.join(data2).encode('latin-1')
    # print(type(joined))
    # print(joined)
    # print(type(frame.read()))
    # print(type(ContentFile(frame.read())))
    path = default_storage.save('tmp/test.wav', ContentFile(frame.read()))
    #print(path)
    blob = request.FILES['blob']

    #path2 = default_storage.save('tmp/blob.wav', ContentFile(blob.read()))

    tmp_file = os.path.join('', path)
    #tmp_file2 = os.path.join('', path2)

    mydata = scipy.io.wavfile.read(tmp_file)

    mydata = mydata[1][:,0]

    if os.path.isfile(tmp_file):
        #os.remove(tmp_file)
        pass




    audiofile = get_audiofile("test.wav",data=mydata,flag=False,frame_size=16000)

    result = cnnPredict(audiofile)
    # Get index of largest probability
    index = np.argmax(result)
    # Get string label from index
    label = index_to_label(index)


    return HttpResponse(json.dumps({'emotion': label}), content_type="application/json")
