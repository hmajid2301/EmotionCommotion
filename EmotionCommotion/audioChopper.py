import scipy.io.wavfile as wav   # Reads wav file
import wave
import sys

import numpy as np
import pandas as pd
import os
from glob import glob
import sys
from types import *
import json

#use soxi -D out.wav, to find length of output in seconds

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

def open_output_file(count, emmotion_number, AUDIOPATH):
	filename = AUDIOPATH + 'out_' + str(count) + "_" + map_emmotion_number(emmotion_number) + ".wav"
	print("\nOpening new output file: " + filename + "\n")
	wave_write = wave.open(filename, 'wb')
	wave_write.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
	#The tuple should be (nchannels, sampwidth, framerate, nframes, comptype, compname)
	return wave_write

def validity_test(triples):
	if (len(triples) == 0):
		return EMPTY_LIST_ERROR 
	for t in range(0,len(triples)):
		this_triple = triples[t]
		if (len(this_triple) != 3):
			print(this_triple)
			return START_TIME_ERROR
		if not (this_triple[2] in range(0,4)):
			return INVALID_EMMOTION_ERROR
	return VALID

def analyse_intervals(triples):
	first_triple = triples[0]
	overall_start_time = first_triple[0]
	last_triple = triples[len(triples)-1]
	overall_finish_time = last_triple[1]
	return overall_start_time, overall_finish_time

def audioChopper(filepath,triples):
	'''
	Triples is a list of startTime/finishTime/emmotion triples. 
	Times are given in seconds.
	Emmotions are represented by a number in the range 0 to 3.
	Example: audioChopper("~/audio/clip_1.wav", [[1,3,0],[4,8,3],[13,20,1]])
	'''

	AUDIOPATH = os.path.dirname(filepath) + '/'

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

	[overall_start_time, overall_finish_time] = analyse_intervals(triples)

	input_file = open(filepath, 'rb')
	[sample_rate, audio] = wav.read(input_file)
	
	sample_rate = sample_rate*4	

	count = 0
	output_file = open_output_file(count, triples[count][2], AUDIOPATH) 
	
	for i in range(overall_start_time, overall_finish_time+1):
		print("Considering second: " + str(i))
		if(triples[count][1] < i):
			output_file.close()
			if(count != len(triples)+1):
				count = count + 1
				output_file = open_output_file(count, triples[count][2], AUDIOPATH)
			else:
				break;
		if(triples[count][0] <= i):
			print("Writing frames from " + str(i) + "s to " + str(i+1) + "s")
			
			#a second worth of frames
			frame_range = range(i*sample_rate, ((i+1)*sample_rate))
			frames = audio[frame_range]
			output_file.writeframes(frames)

	input_file.close()

filepath = sys.argv[1]
triples = json.loads(sys.argv[2])
audioChopper(filepath, triples)
