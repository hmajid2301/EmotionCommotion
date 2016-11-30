#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:04:22 2016

@author: Tom
"""
import sys
sys.path.append('/dcs/project/emotcomm/EmotionCommotion/EmotionCommotion/backend/sourceFiles/')
import thinkplot as tp
import thinkdsp as td
from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

#make_spectrum is sorted in order of decreasing amplitude, so the first pair is going to be the amplitude of the pitch and the pitch frequency.
def pitch(frame, audiofile):
    clip = td.Wave(frame)
    spectrum = clip.make_spectrum()
    return spectrum.peaks()[0][1]    

extractAndSave(pitch,['pitch'],IEMOCAP_LOCATION,2)
