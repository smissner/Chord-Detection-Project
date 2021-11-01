from Chord_Detection import computePCIT

import os
import numpy as np
import scipy as sp
from scipy.io import wavfile

def test(complete_path_to_data_folder):
    searchPath = complete_path_to_data_folder
    rms = []
    for root, dir, files in os.walk(searchPath):
        for waveFile in filter(lambda a: 'wav' in a, files):
            txt = f"{waveFile.split('.')[0]}.f0.Corrected.txt"
            (fs, signal) = wavfile.read(f"{searchPath}/{waveFile}")
            #print(fs)
            #print(len(data))
            arr = []
            txtpath = f"{searchPath}/{txt}"
            with open(txtpath) as file:
                line = file.readline()
                while line:
                    arr.append(np.array([float(x) for x in line.strip().split('     ')]))
                    line = file.readline()
            arr = np.array(arr)

            blocks = 1024
            hops = 1024

            data = computePCIT(signal, blocks, hops, fs)
            """
            ind = 0
            durationOfBlock = (blocks * 1.0/fs)
            estimate = []
            actual = []
            for idx, time in enumerate(timeInSec):
                while(arr[ind][0] < time):
                    ind += 1
                #print(arr[ind][0], time)


                if (arr[ind][0] + arr[ind][1] < time + durationOfBlock):
                    #print(arr[ind][0], arr[ind][0] + arr[ind][1], '<', time, time + durationOfBlock)
                    if (arr[ind][2] > 0 and f0[idx] > 0 ):
                        estimate.append(f0[idx])
                        actual.append(arr[ind][2])
            rms.append(eval_pitchtrack(np.array(estimate), np.array(actual)))
            """
    #print(rms)
    rms = np.array(rms)
    return np.mean(rms)


test('./baselineDataset')
