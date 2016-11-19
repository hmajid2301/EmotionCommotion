# -*- coding: utf-8 -*-
import numpy as np

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