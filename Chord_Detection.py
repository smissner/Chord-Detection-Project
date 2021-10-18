
import scipy as sci
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
def noteName(f,bass,name):
    nAwayFromA = np.mod(np.round(12*np.log2(f/bass)),12)
    note = {
        0: "A",
        1: "Bb",
        2: "B",
        3: "C",
        4: "Db",
        5: "D",
        6: "Eb",
        7: "E",
        8: "F",
        9: "Gb",
        10: "G",
        11: "Ab"
    }
    freqs = dict((v, k) for k, v in note.items())
    newnote = {}
    for i in range(12):
        newnote[i] = note.get(np.mod(freqs.get(name)+i,12))
    return newnote.get(nAwayFromA,"DC")

def computeChromagram(x,blockSize,hopSize,fs):
    f,t,a = sig.spectrogram(x,fs,nperseg = blockSize,noverlap = blockSize - hopSize)
    a = a/np.max(a)
    bassf = f[np.where(a>.1)[0][0]]
    bassnote = noteName(bassf,440,"A")
    notes = noteName(f,bassf,bassnote)
    return bassf


def block_audio(x, blockSize, hopSize, fs):
    if hopSize >= len(x):
        print('hopSize too large')
        return 0

    if blockSize >= len(x):
        print('blockSize too large')
        return 0
    xbList = []  #block array
    timeInSec = np.array([])
    for i in range(0, len(x) - blockSize + 1, hopSize):
        xbList.append(x[i:i + blockSize])
        timeInSec = np.append(timeInSec, (1/fs * i))
    return np.array(xbList), timeInSec


def track_pitch_acf(x,blockSize,hopSize,fs):
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    f0 = np.array([get_f0_from_acf(comp_acf(block, False), fs) for block in xb])

    return (f0, timeInSec) #hz of f0 found here

def run_evaluation(complete_path_to_data_folder):
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

            f0, time = track_pitch_acf(signal, blocks, hops, fs)
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

#CHANGE THIS FOR YOUR COMPUTER
run_evaluation('/Users/andreaspaljug/Documents/gtFall2021/MUSI-Analysis/proj/Chord-Detection-Project/baselineDataset')
