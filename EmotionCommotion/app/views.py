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
from sklearn import preprocessing
import pandas as pd1

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
    #mydata = np.fromfile(open(tmp_file),np.int16)[24:]
    mydata = scipy.io.wavfile.read(tmp_file)
    mydata = mydata[1][:,0]

    audiofile = get_audiofile("test.wav",data=mydata,flag=False)
    features = [amplitude,energy,f0,silence_ratio,zerocrossing,cepstrum,mfcc]
    frames = get_frames(audiofile)
    
    agg_vals = []
    for feature in features:
        vals = []
        for frame in frames:
            vals.append(feature(frame, audiofile))
        vals = np.array(vals)
        agg_vals = np.concatenate((agg_vals,aggregate(vals)), axis=0)
    
    training = pd.read_csv('backend/data/allFeatures.csv')
    training.drop('session',axis=1,inplace=True)
    training = training.replace([np.inf, -np.inf], np.nan)
    training = training.fillna(0)

    training = training.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)


    #agg_vals = agg_vals[0:54]
    agg_vals = np.append(agg_vals[0:18],agg_vals[20:])
    agg_vals = agg_vals.reshape(1,-1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(training)
    agg_vals_scaled = min_max_scaler.transform(agg_vals)
    
   # for a,i in enumerate(training.columns):
   #     print(a,i)
   # for a,i in enumerate(agg_vals_scaled):
   #     print(a,"%.2f" % i)
        
    
    svm = joblib.load('backend/classifiers/svm.pkl') 
    result = svm.predict(agg_vals_scaled)

    if os.path.isfile(tmp_file):
        os.remove(tmp_file)

    return HttpResponse(json.dumps({'emotion': result[0]}), content_type="application/json")
    #return render(request, 'app/index.html', {'data' : result})

   