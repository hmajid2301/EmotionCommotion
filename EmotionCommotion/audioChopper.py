import scipy.io.wavfile as wav   # Reads wav file

import wave
import numpy as np
import pandas as pd
import os
from glob import glob
import sys
from types import *

VALID = 0
TRIPLE_ERROR = 1
START_TIME_ERROR = 2
EMPTY_LIST_ERROR = 3
INVALID_EMMOTION_ERROR = 4

NEUTRAL = 0
HAPPY = 1
ANGRY = 2
SAD = 3

def string_to_list(input):
        output_list = []
        current_sublist = []
        state = 0
        #0: [ not encountered yet
        #1: opening [ just encountered
        #2: [ of a sublist just encountered
        #3: ] of a sublist just encountered
        for i in range(0, len(input)):
                if state == 0:
                        if input[i] == "[":
                                state = 1
                        else:
                                return []
                elif state == 1:
                        if input[i] == "[":
                                state = 2
                        else:
                                return []
                elif state == 2:
                        if input[i] == "]":
                                output_list.append(current_sublist)
                                current_sublist = []
                                state = 3
                        else:
                                if(input[i] != ","):
                                        current_sublist.append(int(input[i]))
                elif state == 3:
                        if input[i] == "[":
                                state = 2
                        else:
                                if((input[i] != ",") & (input[i] != "]")):
                                        return []
        return output_list

def map_emmotion_number(emmotion_number):
        if(emmotion_number == NEUTRAL):
                return "Neutral"
        elif(emmotion_number == HAPPY):
                return "Happy"
        elif(emmotion_number == ANGRY):
                return "Angry"
        else:
                return "Sad"

#Wave_write.setparams(tuple)
#The tuple should be (nchannels, sampwidth, framerate, nframes, comptype, compname), with values valid for the set*() methods. Sets all parameters.

def open_output_file(count, emmotion_number):
        wave_write = wave.open(AUDIOPATH + 'out_' + str(count) + "_" +  map_emmotion_number(emmotion_number)  + ".wav" , 'wb')
        wave_write.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
        return wave_write

def validity_test(triples):
        if (len(triples) == 0):
                return EMPTY_LIST_ERROR
        for t in range(0,len(triples)):
                this_triple = triples[t]
                if (len(this_triple) != 3):
                        print(this_triple)
                        return TRIPLE_ERROR
                if (this_triple[0] > this_triple[1]):
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

        print(triples)

        error_code = validity_test(triples)
        if(error_code == TRIPLE_ERROR):
                print("Error: An input was not a triple")
                return
        if(error_code == START_TIME_ERROR):
                print("Error: An start time was before a finish time")
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

        count = 0
        output_file = open_output_file(count, triples[count][2])

        for i in range(overall_start_time, overall_finish_time+1):
                if(triples[count][1] < i):
                        output_file.close()
                        if(count != len(triples)+1):
                                count = count + 1
                                output_file = open_output_file(count, triples[count][2])
                        else:
                                break;
                if(triples[count][0] <= i):
                        #a second worth of frames
                        frame_range = slice(i*sample_rate, ((i+1)*sample_rate)-1)
                        frames = audio[frame_range]
                        output_file.writeframes(frames)

        input_file.close()

#filepath, triples
filepath = sys.argv[1]
triples = string_to_list(sys.argv[2])
audioChopper(filepath, triples)
