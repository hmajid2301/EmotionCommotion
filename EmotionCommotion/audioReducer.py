import scipy.io.wavfile as wav   # Reads wav file
import sys
import csv
import ntpath

import numpy as np
import pandas as pd
import os
from glob import glob
import sys
from types import *
import json

#AUDIOPATH ON EMOTCOMM: "/dcs/project/emotcomm/local/wild_dataset/chopped_and_labelled/"
MIN_CLIP_LENGTH = 10 #min length of output clips in seconds

def save_clip(start, finish, emmotion, counter, audio, sample_rate):
	samples = audio[start:finish]
	output_name = emmotion + "_" + str(counter) + ".wav"
	wav.write(output_name, sample_rate, samples)
	print("Saved interval from sample " + str(start) + " to sample " + str(finish) + " at " + output_name + "\n")
	
	#add to CSV file
	fields = [output_name,emmotion]
	csv_name = 'emmotion_labels.csv'
	fd = open(csv_name,'a')
	fd.write(output_name+","+emmotion+"\n")
	print("Appened to output csv file: " + csv_name)

def audioReducer(audiopath):
	print("\n")
	print("WARNING: `audioReducer.py' has been written to postprocess the output of `audioChopper.py'. Otherwise ensure that 1. filenames begin *_ 2. include only emotion * 3. are in .wav format [Where * is Happy, Sad, Neutral or Angry]")
	print("\n")
	print("Processing audio at: " + audiopath)
	print("\n")
	
	counter = 0

	for filename in os.listdir(audiopath):
		fullpath = os.path.join(audiopath, filename)
		if filename.endswith(".wav"):
			print("Considering: " + fullpath + "\n")
			
			emmotion = filename.split('_', 1)[0]
			[sample_rate, audio] = wav.read(fullpath)
			
			audio_length = len(audio)/sample_rate
			current_length = audio_length
			sample_position = 0
			
			while(current_length >= MIN_CLIP_LENGTH*2):
				#take MIN_CLIP_LENGTH seconds and write to seperate file
				start = sample_position * sample_rate
				finish = (sample_position + MIN_CLIP_LENGTH) * sample_rate
				save_clip(start, finish, emmotion, counter, audio, sample_rate)
				counter = counter + 1
				
				#move up the sample position so that the same samples aren't written twice
				sample_position = sample_position + MIN_CLIP_LENGTH
				current_length = current_length - MIN_CLIP_LENGTH
				
			#write remaining seconds to seperate file
			start = sample_position * sample_rate
			finish = audio_length * sample_rate
			save_clip(start, finish, emmotion, counter, audio, sample_rate)
			counter = counter + 1
			
		else:
			print("Skipping non-wavfile: " + fullpath + "\n")

audiopath = sys.argv[1]
audioReducer(audiopath)
