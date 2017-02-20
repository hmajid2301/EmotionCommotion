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

#Use soxi -D file_name.wav, to find length of output in seconds

#If you get a 'chunk data' error, try loading your audio into audacity and exporting as windows 16bit floating point WAV

VALID = 0
TRIPLE_ERROR = 1
START_TIME_ERROR = 2
EMPTY_LIST_ERROR = 3
INVALID_EMMOTION_ERROR = 4

NEUTRAL = 0
HAPPY = 1
ANGRY = 2
SAD = 3

def map_emmotion_number(emmotion_number):
	if(emmotion_number == NEUTRAL):
		return "Neutral"
	elif(emmotion_number == HAPPY):
		return "Happy"
	elif(emmotion_number == ANGRY):
		return "Angry"
	else:
		return "Sad"

def validity_test(triples):
	if (len(triples) == 0):
		return EMPTY_LIST_ERROR 
	for t in range(0,len(triples)):
		this_triple = triples[t]
		if (len(this_triple) != 3):
			return START_TIME_ERROR
		if not (this_triple[2] in range(0,4)):
			return INVALID_EMMOTION_ERROR
	return VALID

def audioChopper(filepath,triples):
	'''
	Triples is a list of startTime/finishTime/emmotion triples. 
	Times are given in seconds.
	Emmotions are represented by a number in the range 0 to 3.
	Example: audioChopper("~/audio/clip_1.wav", [[1,3,0],[4,8,3],[13,20,1]])
	'''

	print("\nINPUT TRIPLES (Start seconds, finish seconds, emotion number):")
	print(triples)
	print("\n")

	error_code = validity_test(triples)
	if(error_code == TRIPLE_ERROR):
		print("Error: An input was not a triple")
		return
	if(error_code == START_TIME_ERROR):
		print("Error: A start time was before a finish time")
		return
	if(error_code == EMPTY_LIST_ERROR):
		print("Error: Please give at least one interval")
		return
	if(error_code == INVALID_EMMOTION_ERROR):
		print("Error: Invalid emmotion number given")
		return

	#input_file = open(filepath, 'rb')
	[sample_rate, audio] = wav.read(filepath)
	print("sample rate = " + str(sample_rate))
	sample_rate = sample_rate
	
	#len(audio)/sample_rate=length of audio in seconds	
	for i in range(len(triples)):
		t = triples[i]
		start = t[0]*sample_rate
		finish = (t[1]+1)*sample_rate
		samples = audio[start:finish]
		emmotion = map_emmotion_number(t[2])
		output_name = emmotion + "_" + str(i) + "_" + ntpath.basename(filepath)
		wav.write(output_name, sample_rate, samples)
		print("Saved interval from sample " + str(start) + " to sample " + str(finish) + " at " + output_name)
		
		fields = [output_name,emmotion]
		csv_name = 'emmotion_labels.csv'
		fd = open(csv_name,'a')
		fd.write(output_name+","+emmotion+"\n")
		print("Appened to output csv file: " + csv_name)

filepath = sys.argv[1]
triples = json.loads(sys.argv[2])
audioChopper(filepath, triples)
