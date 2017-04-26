import sys
sys.path.append('../../app')
sys.path.append('../')

import allExtractors as ext
from datagrabber import get_frames,get_audiofile,aggregate
import time
import matplotlib.pyplot as plt
import matplotlib

# List of feature extractor functions
extractors = [ext.amplitude,ext.cepstrum,ext.energy,ext.f0,ext.mfcc_with_rounding,ext.silence_ratio,ext.zerocrossing]

# Read a 10 second clip for testing
audiofile = get_audiofile('10_sec_clip.wav')
frames = get_frames(audiofile)
results = {}

# Extract each feature and store time
for funct in extractors:
	start = time.time()
	vals = []
	for frame in frames:
	    vals.append(funct(frame, audiofile))
	agg_vals = aggregate(vals)
	end = time.time()

	results[funct.__name__] = end - start

# Print total extraction time
print("Total time:" + str(sum(results.values())))

# Figure aesthetics
font = {'family' : 'normal',
		'size'   : 18}
matplotlib.rc('font', **font)

# Plot extraction times
plt.bar(range(len(results)), results.values(), align='center')
plt.xticks(range(len(results)), results.keys())
plt.ylabel('Extraction time (seconds)')
plt.xlabel('Feature')

plt.show()
