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
import allExtractors as ext
import datagrabber as dg




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
    os.rename("C:\\Users\\Haseeb Majid\\Downloads\\test.wav", "test.wav")
    data = np.fromfile(open('test.wav'),np.int16)[24:]
    print(data)
    
    audiofile = dg.get_audiofile(filename="test.wav",data=data)
    
    
    features = [ext.amplitude,ext.cepstrum,ext.energy,ext.silence_ratio,ext.zerocrossing]

    frames = dg.get_frames(audiofile)
    
    agg_vals = []
    for feature in features:
        vals = []
        for frame in frames:
            vals.append(feature(frame, audiofile))
        agg_vals = np.concatenate((agg_vals,dg.aggregate(vals)), axis=0)
        
    
    svm = joblib.load('../backend/classifiers/svm.pkl') 
    print(svm.predict(agg_vals))

    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )