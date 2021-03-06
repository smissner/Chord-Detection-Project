from Chord_Detection import computePCIT
from Chord_Detection import findchordnotes
from Chord_Detection import correlateChords
from onset_detection import get_onsets

import os
import numpy as np
import scipy as sp
from scipy.io import wavfile
import music21 as m

## this section is for chordify  / mcgill billboard integration, put on the backburner for now

"""
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
"""


#This section is for integration with isophonics dataset.  Primary section now.
## http://isophonics.net/content/reference-annotations

def parse_isophonics_csv(filename):
    splitC = ' '
    times = []
    chords = []
    with open(filename) as file:
        line = file.readline()
        while line:
            arr = line.strip().split(splitC)
            r = np.array([float(i) for i in arr[0:2]])
            chords.append(arr[2])
            #print(arr[2])
            times.append(r)
            line = file.readline()

    times = np.array(times)
    chords = np.array(chords)
    return (times, chords)

def compareannotationswithpcp(annots, pcp):
    #print(annots)
    #print(pcp)
    #pcp[pcp <= .4] = 0
    values, weights= findchordnotes(pcp, 5)

    chords = {}
    curr = []
    for i in range(len(weights)):
        indx = np.argmax(weights)
        curr += values[indx]
        #print(weights)
        chd = m.chord.Chord(curr) # swap this out for correlation function
        chd.simplifyEnharmonics(inPlace = True)
        chd.sortAscending(inPlace  = True)
        name = chd.pitchedCommonName
        values = np.delete(values, indx)
        weights = np.delete(weights, indx)
        if i >= 3:
            if (name in chords):
                chords[name] += 1
            else:
                chords[name] = 1
    print(annots, "\t", chords)

def compareannotationswithpcpwithcustomchorddetect(annots, pcp, sevenths):
    #print(annots)
    #print(pcp)
    #pcp[pcp <= .4] = 0
    #values, weights= findchordnotes(pcp, 5)
    result = correlateChords(pcp, sevenths)
    #print(annots, result)
    return result
    #print(annots, "\t", chords)

def check_chord_name(result ):
    print()

def evaluate_isophonics_wav(csvF, wfilename, sevenths=True):
    #print('starting next song', csvF, wfilename)
    (fs, signal) = wavfile.read(wfilename)
    #print(fs)
    #print(len(data))
    #print(signal)
    signal = np.mean(signal, axis=1)

    # trying normalizing audio data
    signal=signal/np.mean(signal)
    #print(signal)
    (antimes, chords) = parse_isophonics_csv(csvF)

    on_detect_hop=256
    on_detect_frame=512
    onset_thresh=.7
    blocks, peak_array = get_onsets(signal, fs, on_detect_hop, on_detect_frame, onset_thresh) # returns optimal block length and peak locations in samples
        #      get_onsets(x, fs, hop_length = 256, frame_length = 512, onset_thresh=.75)
    hops = blocks/2

    [pcp, pcpt] = computePCIT(signal, blocks, hops, peak_array, fs)
    print('annotated data: ')
    #print(chords)
    #print('projected data: ')

    printtime = False
    idxC = 0;
    #count = 0

    accuracyBase = 0
    accuracyTriadQuality = 0
    totalBaseEvaluated = 0
    totalTriadsQualEvaluated = 0
    everything = 0
    everythinTotalEvaluated = 0
    for idxT, time in enumerate(pcpt):

        #print(f"A1: time: {         time}, \t annotated time: {    antimes[idxC][0]}")


        if idxC < len(antimes) and time >= antimes[idxC][0] and time <= antimes[idxC][1]:
            #if (printtime):
                #print(f"A1: res time: { time}, \t annotated time window: {    antimes[idxC]}")
            result = compareannotationswithpcpwithcustomchorddetect(chords[idxC], pcp[idxT], sevenths)



            #print(f"A2: res val: {  pcp[idxT]}, \t annotated val: {antimes[idxC]}")
            #check result here!
            if result != 'N':
                baseRes, baseQual = result.split(':')
                splitChord = chords[idxC].split(':')
                baseChord = splitChord[0]

                if baseChord != 'N':
                    triadCorrect = False
                    if len(splitChord) > 1:
                        quality = splitChord[1].split('/')[0] #remove the bass note annotations
                        if not quality.isdigit():
                            if (baseQual.lower() == quality):
                                #print("MATCH:",baseQual.lower())
                                accuracyTriadQuality += 1
                                triadCorrect = True
                            totalTriadsQualEvaluated += 1
                        else:
                            if (quality == baseQual):
                                print("MATCH:",baseQual)
                                accuracyTriadQuality += 1
                                triadCorrect = True
                            totalTriadsQualEvaluated += 1

                    else:
                        if (baseQual.lower() == 'maj'):
                            #print("MATCH:",baseQual.lower())
                            accuracyTriadQuality += 1
                            triadCorrect = True
                        totalTriadsQualEvaluated += 1

                    if (baseChord == baseRes):
                        #print("MATCH:",baseChord)
                        accuracyBase += 1
                        if (triadCorrect):
                            everything += 1
                    elif (baseChord =='A#' and baseRes == "Bb"):
                        accuracyBase += 1
                        if (triadCorrect):
                            everything += 1
                        #A# Bfl,
                    elif (baseChord =='C#' and baseRes == "Db"):
                        accuracyBase += 1
                        if (triadCorrect):
                            everything += 1
                        #c# dflat,
                    elif (baseChord =='D#' and baseRes == "Eb"):
                        accuracyBase += 1
                        if (triadCorrect):
                            everything += 1
                        #d#, e flat,
                    elif (baseChord =='F#' and baseRes == "Gb"):
                        accuracyBase += 1
                        if (triadCorrect):
                            everything += 1
                        #f#, g flat,
                    elif (baseChord =='G#' and baseRes == "Ab"):
                        accuracyBase += 1
                        if (triadCorrect):
                            everything += 1
                        #g#, a flat,

                    totalBaseEvaluated += 1
                    everythinTotalEvaluated += 1

        tempidxC = idxC
        while (idxC < len(antimes) and time >= antimes[idxC][1]):
            #print(idxC)
            idxC += 1
        if tempidxC != idxC and idxC >= 0 and idxC < len(antimes) and time >= antimes[idxC][0] and time <= antimes[idxC][1]:
        #    print(count)
        #    count = 0
            if (printtime):
                print(f"B1: res time: { time}, \t annotated time: {    antimes[idxC]}")
            #print(f"B2: res val: {  pcp[idxT]}, \t annotated val: { antimes[idxC]}")
            #check result here!
        #else:
        #    count += 1
    everythingAvg, averageBase, averageQual = 0, 0, 0
    if (totalBaseEvaluated != 0):
        averageBase = accuracyBase / totalBaseEvaluated
    if (totalTriadsQualEvaluated != 0):
        averageQual = accuracyTriadQuality / totalTriadsQualEvaluated

    if (everythinTotalEvaluated != 0):
        everythingAvg = everything /everythinTotalEvaluated
    print(everythingAvg, averageBase, averageQual)
    return [accuracyBase, totalBaseEvaluated, accuracyTriadQuality, totalTriadsQualEvaluated, everything, everythinTotalEvaluated]

txt = './chords/stl.txt'
wav = './chords/stl.wav'
#evaluate_isophonics_wav(txt, wav)


def beatles_check_album(searchPath):
    anont = "./QMUL_beatles/C4DM_beatles_transcriptions"
    albumStem = searchPath.split('/')[-1]
    print(albumStem)
    accuracy = [0]*6
    for root, dir, files in os.walk(searchPath):
        for waveFile in filter(lambda a: 'wav' in a, files):
            songname = waveFile.split('.')[0]
            print(f"begining {songname}")
            songname = songname.replace('-', '_-_', 1)

            temp_acc =  evaluate_isophonics_wav(f"{anont}/{albumStem}/{songname}.lab", f"{searchPath}/{waveFile}")
            for i, a in enumerate(temp_acc):
                accuracy[i] += a

    everythingAvg, averageBase, averageQual = 0, 0, 0
    if (accuracy[1] != 0):
        averageBase = accuracy[0] / accuracy[1]
    if (accuracy[3] != 0):
        averageQual = accuracy[2] / accuracy[3]
    if (accuracy[5] != 0):
        everythingAvg = accuracy[4] / accuracy[5]
    print(everythingAvg, averageBase, averageQual)
    return everythingAvg, averageBase, averageQual
#searchPath = "./QMUL_beatles/01_-_Please_Please_Me"
searchPath = "./QMUL_beatles/11_-_Abbey_Road"
#searchPath = "./QMUL_beatles/08_-_Sgt_Peppers_Lonely_Hearts_Club_Band"
#searchPath = "./QMUL_beatles/07_-_Revolver"


beatles_check_album(searchPath)

def check_all_albums():
    accuracy = [0]*6
    base = './QMUL_beatles'
    for root, dir, files in os.walk(base):
        #albumStem = searchPath.split('/')[-1]
        print(root)
        for waveFile in filter(lambda a: 'wav' in a, files):
            songname = waveFile.split('.')[0]
            songname = songname.replace('-', '_-_', 1)
            temp_acc = evaluate_isophonics_wav(f"./QMUL_beatles/C4DM_beatles_transcriptions/{root.split('/')[2]}/{songname}.lab", f"{root}/{waveFile}")
            for i, a in enumerate(temp_acc):
                accuracy[i] += a

    everythingAvg, averageBase, averageQual = 0, 0, 0
    if (accuracy[1] != 0):
        averageBase = accuracy[0] / accuracy[1]
    if (accuracy[3] != 0):
        averageQual = accuracy[2] / accuracy[3]
    if (accuracy[5] != 0):
        everythingAvg = accuracy[4] / accuracy[5]
    print(everythingAvg, averageBase, averageQual)
    return everythingAvg, averageBase, averageQual



#check_all_albums()

#def test2():


#test('./baselineDataset')


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
