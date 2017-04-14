import sys
sys.path.append('../../app')
sys.path.append('../')

import allExtractors as ext
from datagrabber import get_frames,get_audiofile,aggregate
import time
import matplotlib.pyplot as plt
import matplotlib

extractors = [ext.amplitude,ext.cepstrum,ext.energy,ext.f0,ext.mfcc_with_rounding,ext.silence_ratio,ext.zerocrossing]


audiofile = get_audiofile('10_sec_clip.wav')
frames = get_frames(audiofile)
results = {}

for funct in extractors:
	start = time.time()
	vals = []
	for frame in frames:
	    vals.append(funct(frame, audiofile))
	agg_vals = aggregate(vals)
	end = time.time()

	results[funct.__name__] = end - start

font = {'family' : 'normal',
		'size'   : 18}

matplotlib.rc('font', **font)


plt.bar(range(len(results)), results.values(), align='center')
plt.xticks(range(len(results)), results.keys())
plt.ylabel('Extraction time (seconds)')
plt.xlabel('Feature')

plt.show()