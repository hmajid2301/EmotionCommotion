#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:04:22 2016

@author: Tom
"""
from importlib.machinery import SourceFileLoader

# assuming for now that thinkdsp and clip is in the same directory
# the energy function returns the sum of the amplitudes squared (as energy is proportional to amplitude squared)
def energy(dirPath, clipName):
    thinkplot = SourceFileLoader("thinkplot", dirPath + "thinkplot.py").load_module()
    thinkdsp = SourceFileLoader("thinkdsp", dirPath + "thinkdsp.py").load_module()

    clip = thinkdsp.read_wave(dirPath + clipName)

    ###shortening the clip is a good idea for testing###
    #start = 1.2
    #duration = 0.6
    #clip = clip.segment(start, duration)

    spectrum = clip.make_spectrum()
    # there are ALOT of peaks in a clip so this may be really inefficient.
    # some sort of threshold may be useful for this function too, if only to speed it up, see how fast it performs first!

    peaks = spectrum.peaks()

    totalEnergy = 0
    N = len(peaks) - 1
    for index in range(0, N):
        totalEnergy = totalEnergy + peaks[index][0] ** 2

    return totalEnergy


energy('C:/Users/Tom/PycharmProjects/SoundTest/ThinkDSP-master/code/', '92002__jcveliz__violin-origional.wav')

##https://www.boundless.com/physics/textbooks/boundless-physics-textbook/waves-and-vibrations-15/wave-behavior-and-interaction-126/energy-intensity-frequency-and-amplitude-450-1130/
#energy is proportional to amplitude^2, frequency generally assumed to not impact energy