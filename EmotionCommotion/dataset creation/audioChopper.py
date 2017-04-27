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

#constants for error codes
VALID = 0
TRIPLE_ERROR = 1
START_TIME_ERROR = 2
EMPTY_LIST_ERROR = 3
INVALID_EMOTION_ERROR = 4

#constants for emotions
NEUTRAL = 0
HAPPY = 1
ANGRY = 2
SAD = 3

#function to take a number from 0 to 3 as an input and output the emotion as a string
def map_emotion_number(emotion_number):
	if(emotion_number == NEUTRAL):
		return "Neutral"
	elif(emotion_number == HAPPY):
		return "Happy"
	elif(emotion_number == ANGRY):
		return "Angry"
	else:
		return "Sad"

#function which carries out a series of checks then returns an error code
def validity_test(triples):
	if (len(triples) == 0):
		return EMPTY_LIST_ERROR 
	for t in range(0,len(triples)):
		this_triple = triples[t]
		if (len(this_triple) != 3):
			return TRIPLE_ERROR
		if (this_triple[0] > this_triple[1]):
			return START_TIME_ERROR
		if not (this_triple[2] in range(0,4)):
			return INVALID_EMOTION_ERROR
	return VALID

def audioChopper(filepath,triples):
	'''
	'triples' is a list of startTime/finishTime/emotion triples. 
	Times are given in seconds.
	Emotions are represented by a number in the range 0 to 3.
	Example: audioChopper("~/audio/clip_1.wav", [[1,3,0],[4,8,3],[13,20,1]])
	'''

	print("\nINPUT TRIPLES (Start seconds, finish seconds, emotion number):")
	print(triples)
	print("\n")

	#check for errors, return and print an error message if an error is found
	error_code = validity_test(triples)
	if(error_code == TRIPLE_ERROR):
		print("Error: An input was not a triple")
		return
	if(error_code == START_TIME_ERROR):
		print("Error: A start time was after a finish time")
		return
	if(error_code == EMPTY_LIST_ERROR):
		print("Error: Please give at least one interval")
		return
	if(error_code == INVALID_EMOTION_ERROR):
		print("Error: Invalid emotion number given")
		return
	
	#wav.read gets the sample rate and creates an audio variable which allows use of the audio data at filepath
	[sample_rate, audio] = wav.read(filepath)
	print("sample rate = " + str(sample_rate))
	
	#len(audio)/sample_rate=length of audio in seconds
	
	#for loop to iterate over each triple
	for i in range(len(triples)):
		#t = the next triple to consider
		t = triples[i]
		#isolate the samples of interest from the original audio clip
		start = t[0]*sample_rate
		finish = (t[1]+1)*sample_rate
		samples = audio[start:finish]
		#map the input emotion number to a string
		emotion = map_emotion_number(t[2])
		#construct the file name for the generated file
		output_name = emotion + "_" + str(i) + "_" + ntpath.basename(filepath)
		#write the samples to a new file
		wav.write(output_name, sample_rate, samples)
		print("Saved interval from sample " + str(start) + " to sample " + str(finish) + " at " + output_name)
		
		#append the name of the new file and its emotion to a CSV file
		fields = [output_name,emotion]
		csv_name = 'emotion_labels.csv'
		fd = open(csv_name,'a')
		fd.write(output_name+","+emotion+"\n")
		print("Appened to output csv file: " + csv_name)

filepath = sys.argv[1]
triples = json.loads(sys.argv[2]) #json.loads allows command line input to be treated as a python list
audioChopper(filepath, triples)
