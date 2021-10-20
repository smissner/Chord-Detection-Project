from Chord_Detection import computePCIT

import os
import numpy as np
import scipy as sp
from scipy.io import wavfile
"""
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

    #print(rms)
    rms = np.array(rms)
    return np.mean(rms)
"""
"""
from rdflib import Graph

g = Graph()
g.parse("01 Bohemian Rhapsody.ttl",  format='ttl')

import pprint
for stmt in g:
    pprint.pprint(stmt)

"""

## this section is for
import jams
import subprocess
from pytube import YouTube
import os

def download_wav(filename):
    jam = jams.load(filename)
    link = jam['file_metadata']['identifiers']['youtube_url']

    #Youtube pull
    print(link)
    yt = YouTube(link)

    video = yt.streams.filter(only_audio=True).first()

    out_file = video.download(output_path=".")

    base, ext = os.path.splitext(out_file)
    new_file = base.translate({' ': None }) + '.wav'
    #os.rename(out_file, new_file)

    subprocess.call(['ffmpeg', '-i', out_file ,
                       new_file])

fl = '12.jams'
#download_wav(fl)


## http://isophonics.net/content/reference-annotations

def parse_isophonics_csv(filename):
    times = []
    chords = []
    with open(filename) as file:
        line = file.readline()
        while line:
            arr = line.strip().split('\t')
            r = np.array([float(i) for i in arr[0:2]])
            chords.append(arr[2])
            #print(arr[2])
            times.append(r)
            line = file.readline()

    times = np.array(times)
    chords = np.array(chords)
    return (times, chords)

def evaluate_isophonics_wav(csvF, wfilename):

    (fs, signal) = wavfile.read(wfilename)
    #print(fs)
    #print(len(data))
    (times, chords) = parse_isophonics_csv(csvF)

    blocks = 1024
    hops = blocks

    data = computePCIT(signal, blocks, hops, fs)
    print(data)

txt = 'stl.txt'
wav = 'stl.wav'
evaluate_isophonics_wav(txt, wav)

#def test2():


#test('./baselineDataset')
