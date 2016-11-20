"""
Definition of views.
"""

from django.shortcuts import render
from django.http import HttpRequest
from django.http import HttpResponse
from django.template import RequestContext
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from sklearn.externals import joblib
from django.shortcuts import render_to_response

import numpy as np
import os
import json


import sys
#sys.path.append("/home/olly/cs/4_year/project/EmotionCommotion/EmotionCommotion")
sys.path.append("./")

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
    data = request.FILES['data'] 
    path = default_storage.save('tmp/test.wav', ContentFile(data.read()))
    tmp_file = os.path.join('', path)
    mydata = np.fromfile(open(tmp_file),np.int16)[24:]

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
        
    
    ##svm = joblib.load('/home/olly/cs/4_year/project/EmotionCommotion/EmotionCommotion/backend/classifiers/svm.pkl') 
    svm = joblib.load('backend/classifiers/svm.pkl') 
    result = svm.predict(agg_vals)
    print(result)

    if os.path.isfile(tmp_file):
        os.remove(tmp_file)

    return HttpResponse(json.dumps({'emotion': result[0]}), content_type="application/json")
    #return render_to_response('app/index.html', data)
    #return render(request, 'app/index.html', {'data' : result})

   