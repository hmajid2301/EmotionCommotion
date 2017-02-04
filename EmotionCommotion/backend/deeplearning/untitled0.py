#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:40:13 2017

@author: olly
"""

import numpy as np
import pandas as pd
from glob import glob
a = np.load("./spectograms/Ses01F_impro01_F001_specto.npy")
labels = pd.read_csv("allLabels.csv")

specs = []
for filename in glob('./spectograms/*.npy'):
    name = filename.split('/')[-1][:-4]
    specs.append(np.load(filename))