#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:04:22 2016

@author: Tom
"""
#from importlib.machinery import SourceFileLoader

import sys
sys.path.append('../sourceFiles/')
sys.path.append('../')
import thinkplot as tp
import thinkdsp as td
from datagrabber import extractAndSave,extractAndSaveYoutubeData

#datset filepaths
IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"

threshold=0.5
#assuming for now that thinkdsp and clip is in the same directory
def f0(frame, audiofile):
    #create a 'spectrum' list for the frame consisting of frequency/amplitude pairs
    clip = td.Wave(frame)
    spectrum = clip.make_spectrum()

    # to find F0 we can sort by Hz, unfortunately we have small readings at 0.0Hz
    # and other small frequencies which make finding the true F0 hard
    # remove the small amplitudes which don't well describe the energy in the sound.
    
    # What is small is likely to be relative to the largest amplitude so use some threshold
    threshold=0.5
    
    #the pairs are sorted by amplitude, take the amplitude of the first pair as the greatest amplitude.
    greatest_amp = (spectrum.peaks()[0])[0]
    #a counter will be used to iterate over the positions of the list
    list_pos = 0
    #the pairs with amplitudes which are within the threshold will be copied to a new list
    selected_pairs = []
    while True:
        #select the next pair from the list
        pair = (spectrum.peaks()[list_pos])
        #test whether the pair's amplitude is within the threshold
        within_threshold = (pair[0] > (greatest_amp * threshold))
        #if not within the threshold the loop can break as all followings pairs will have lower amplitudes
        if within_threshold == False:
            break
        #if within the threshold add the pair to the list
        selected_pairs.append(pair)
        #increment the counter for the loop
        list_pos = list_pos + 1

    # selected_pairs is now a list of the pairs which have a prominent amplitude
    sorted_by_Hz = sorted(selected_pairs, key=lambda tup: tup[1])
    # the lowest Hz should now be F0
    if len(sorted_by_Hz) == 0:
        return [0]
    else:
        return [sorted_by_Hz[0][1]]

# Extract f0 from IEMOCAP and YouTube datasets
extractAndSave(f0,['f0'],IEMOCAP_LOCATION,2,True,True)
extractAndSaveYoutubeData(f0,["f0"],YOUTUBE_LOCATION,2,True,True)
