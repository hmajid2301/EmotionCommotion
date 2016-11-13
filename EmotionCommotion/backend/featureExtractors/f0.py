#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:04:22 2016

@author: Tom
"""
from importlib.machinery import SourceFileLoader
from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"


#assuming for now that thinkdsp and clip is in the same directory
def f0(filename, threshold=0.5):
   # thinkplot = SourceFileLoader("thinkplot", dirPath + "thinkplot.py").load_module()
    thinkdsp = SourceFileLoader("thinkdsp", "../sourceFiles/thinkdsp.py").load_module()

    clip = thinkdsp.read_wave(filename)
    spectrum = clip.make_spectrum()

    #sorted_by_Hz = sorted(spectrum.peaks(), key=lambda tup: tup[1])
    # to find F0 we can sort by Hz, unfortunately we have small readings at 0.0Hz
    # and other small frequencies which make finding the true F0 hard

    # remove the small amplitudes which don't well describe the energy in the sound.
    # What is small is likely to be relative to the largest amplitude so use some threshold
    greatest_Hz = (spectrum.peaks()[0])[0]
    selected_pairs = []
    list_pos = 0
    while True:
        pair = (spectrum.peaks()[list_pos])
        within_threshold = (pair[0] > (greatest_Hz * threshold))
        if within_threshold == False:
            break
        selected_pairs.append(pair)
        list_pos = list_pos + 1

    # selected_pairs is now a list of the pairs which have a prominent amplitude
    sorted_by_Hz = sorted(selected_pairs, key=lambda tup: tup[1])
    # the lowest Hz should now be F0, will need to tweek 0.5
    return sorted_by_Hz[0][0]

f0('/home/olly/cs/4_year/project/local/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro02/Ses01F_impro02_F001.wav',0.5)

extractAndSave(f0,IEMOCAP_LOCATION,2)
