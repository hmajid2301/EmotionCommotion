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

def save_clip(start, finish, emotion, counter, audio, sample_rate):
	#isolate the samples from the audio file to be saved to a new location
	samples = audio[start:finish]
	#construct the output name from the emotion name and counter
	output_name = emotion + "_" + str(counter) + ".wav"
	#write the isolated samples to a new file of name 'output_name'
	wav.write(output_name, sample_rate, samples)
	print("Saved interval from sample " + str(start) + " to sample " + str(finish) + " at " + output_name + "\n")
	
	#add to CSV file
	fields = [output_name,emotion]
	csv_name = 'emotion_labels.csv'
	fd = open(csv_name,'a')
	fd.write(output_name+","+emotion+"\n")
	print("Appened to output csv file: " + csv_name)

def audioReducer(audiopath):
	print("\n")
	print("WARNING: `audioReducer.py' has been written to postprocess the output of `audioChopper.py'. Otherwise ensure that 1. filenames begin *_ 2. include only emotion * 3. are in .wav format [Where * is Happy, Sad, Neutral or Angry]")
	print("\n")
	print("Processing audio at: " + audiopath)
	print("\n")
	
	#counter to ensure each file created has a different name, it must be incremented after save_clip is used
	counter = 0
	
	for filename in os.listdir(audiopath):
		#construct full path from audiopath (directory path) and filename
		fullpath = os.path.join(audiopath, filename)
		#only consider .wav files
		if filename.endswith(".wav"):
			print("Considering: " + fullpath + "\n")
			
			#use split function to select characters before the first underscore (uses assumption that files were created by audioChopper.py)
			emotion = filename.split('_', 1)[0]
			#wav.read creates an audio variable storing the audio data at the fullpath and gets its sample rate
			[sample_rate, audio] = wav.read(fullpath)
			
			#audio_length is the total length of the original file
			audio_length = len(audio)/sample_rate
			#current_length is the length of the original file which has not yet been processed
			current_length = audio_length
			#sample_position is the number of seconds into the original file which have been processed so far
			sample_position = 0
			
			#repeat until there are less than twice the minumum clip length seconds remaining to be processed
			while(current_length >= MIN_CLIP_LENGTH*2):
				#take MIN_CLIP_LENGTH seconds and write to seperate file
				start = sample_position * sample_rate
				finish = (sample_position + MIN_CLIP_LENGTH) * sample_rate
				save_clip(start, finish, emotion, counter, audio, sample_rate)
				counter = counter + 1
				
				#move up the sample position so that the same samples aren't written twice
				sample_position = sample_position + MIN_CLIP_LENGTH
				#the remaining seconds still to be processed has decreased by the minimum clip length
				current_length = current_length - MIN_CLIP_LENGTH
				
			#write remaining seconds to seperate file
			start = sample_position * sample_rate
			finish = audio_length * sample_rate
			save_clip(start, finish, emotion, counter, audio, sample_rate)
			counter = counter + 1
			
		else:
			print("Skipping non-wavfile: " + fullpath + "\n")

audiopath = sys.argv[1]
audioReducer(audiopath)
