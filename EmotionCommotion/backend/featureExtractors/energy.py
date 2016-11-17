#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:04:22 2016

@author: Tom
"""
from datagrabber import extractAndSave
import numpy as np

IEMOCAP_LOCATION = "../../../../local"

def energy(frame):
    return [sum(np.apply_along_axis(lambda x: x**2,0,frame))]

extractAndSave(energy,IEMOCAP_LOCATION,2)