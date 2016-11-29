# -*- coding: utf-8 -*-
import numpy as np
import aubio as aub
import sys
sys.path.append('backend/sourceFiles/')
import thinkdsp as td


def zerocrossing(frame, audiofile):
    i = 0
    for x in range(0, len(frame)-1):
        if (frame[i]*frame[i+1] < 0):
            i += 1
    return [i]

def silence_ratio(frame, audiofile):
    threshold = audiofile['threshold']
    thresholded_frame = frame[np.where(abs(frame) > threshold)]
    ratio = 1 - (len(thresholded_frame) / len(frame))
    return [ratio]

def energy(frame, audiofile):
    return [sum(np.apply_along_axis(lambda x: x**2,0,frame))]

def cepstrum(frame, filename):
    audio = np.fft.fft(frame)
    audio = abs(audio)
    audio = audio ** 2
    audio = np.log2(audio)
    audio = np.fft.ifft(audio)
    return [np.amax(audio), np.average(audio), np.var(audio)]
            
def amplitude(frame, audiofile):
    return [np.amax(frame), np.average(frame),np.var(frame)]
            
            


def mfcc(frame, audiofile):
    coefficientsCount = 12
    
    sampleRate = audiofile['sample_rate']
    frame_size = audiofile['frame_size']
    m = aub.mfcc(frame_size*4, 40, coefficientsCount, sampleRate)
    p = aub.pvoc(frame_size*4, int(frame_size))
    if len(frame) != 128:
        frame = np.pad(frame,(0,frame_size-len(frame)),'constant',constant_values=0)
    spec = p(frame.astype(np.float32))
    
    mfcc_out = m(spec)
    return mfcc_out
    
#assuming for now that thinkdsp and clip is in the same directory
def f0(frame, audiofile):
    threshold=0.5


    clip = td.Wave(frame)
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
    return [sorted_by_Hz[0][1]]