# -*- coding: utf-8 -*-

from scipy.io.wavfile import read
from pylab import plot,show,subplot,specgram
from scipy import signal
import matplotlib.pyplot as plt
rate,data = read('../local/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F010.wav') # reading
rate2,data2 = read('../local/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F011.wav') # reading

a = specgram(data, NFFT=320, noverlap=160) # small window
b = specgram(data2, NFFT=320, noverlap=160) # small window
