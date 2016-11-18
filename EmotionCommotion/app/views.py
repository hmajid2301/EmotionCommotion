"""
Definition of views.
"""

from django.shortcuts import render
from django.http import HttpRequest
from django.template import RequestContext
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import os
from sklearn.externals import joblib

import sys
sys.path.append("/home/olly/cs/4_year/project/EmotionCommotion/EmotionCommotion")

from .datagrabber import *
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
    audio = request.POST['audio-path']

    if os.path.isfile("test.wav"):
        os.remove("test.wav")
    os.rename("/home/olly/Downloads/test.wav", "test.wav")
    mydata = np.fromfile(open('test.wav'),np.int16)[24:]

    audiofile = get_audiofile("test.wav",data=mydata,flag=False)
    

    features = [amplitude,cepstrum,energy,silence_ratio,zerocrossing]

    frames = get_frames(audiofile)
    
    agg_vals = []
    for feature in features:
        vals = []
        for frame in frames:
            vals.append(feature(frame, audiofile))
        vals = np.array(vals)
        agg_vals = np.concatenate((agg_vals,aggregate(vals)), axis=0)
        
    
    svm = joblib.load('/home/olly/cs/4_year/project/EmotionCommotion/EmotionCommotion/backend/classifiers/svm.pkl') 
    print(svm.predict(agg_vals))

    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )