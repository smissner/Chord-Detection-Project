
#Gonna throw everything important for interfacing purposes up here. All the functions here are compiled together into
#that final, computePCIT(Pitch Classes In Time) function. This function will input an audio information as well as the desired
#block and hop sizes. The output will be 1) An array of blocks where each block contain the four most prevalent notes as type
#numpy_str, 2) An array of blocks where each value in the block corresponds to the individual prevalence of their corresponding
#notes, and 3) An array of timestamps for each block.


#An example of an output would be pc,pcp,t: Where pc[0] could be ['G', 'E', 'C', 'B'] and pcp[0] could be [1.,0.80,0.77,0.11]
#This would tell us that, at the first block, G E C and B are the four most prevalent notes, but the G E and C are significantly
#more prevalent than the B, and so we would probably want to ignore that B(I found .2 to be a pretty decent cutoff for significance)
#So, we would want our chord reader to think of this as a Cmaj chord without a 7th, which is what I inputted to get this

import scipy as sci
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
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
def noteName(f,bass,name):
    #This function will get the name of a note given a frequency as well as information pertaining to
    #what our bass note frequency is
    nAwayFromA = np.mod(np.round(12*np.log2(f/bass)),12)
    freqs = dict((v, k) for k, v in note.items())
    newnote = {}
    for i in range(12):
        newnote[i] = note.get(np.mod(freqs.get(name)+i,12))
    return (newnote.get(nAwayFromA,"zDC"))
def computeChromagram(x,blockSize,hopSize,fs):
    #Take the spectrogram of our audio data, and normalize it
    f,t,a = sig.spectrogram(x,fs,nperseg = blockSize,noverlap = blockSize - hopSize)
    a = a/np.max(a)
    #Create a tuning bassline given the lowest prevalent note in the audio. This process seems to be helpful, but only very
    #slightly I've found, and due to possible complications this process creates, we may want to remove it.
    bassf = f[np.where(a>.1)[0][0]]
    if bassf <= .1:
        #if np.where(a>.1).size < 2:
        bassf = .001
        #else: ##Leaving this out for now, will work on it later, this for now shouldn't cause too much issue
         #       bassf = f[np.where(a>.1)[1][0]]
    bassnote = noteName(bassf,440,"A")
    notes = np.array([])
    #Create a new note array to replace the frequency array so that we have a chromatic representation of the spectrogram
    for i in range(f.size):
        tempnote = noteName(f[i],bassf,bassnote)
        notes = np.append(notes,tempnote)
        #This qFactor will hopefully work to make out of tune notes not mess with our data(say someone in the recording plays
        #a really sharp c, we don't want that being recorded as a Db). Works by reducing the value of notes in frequency bins
        #far from a centralized note

        qFactor = 1 - abs(12*np.log2(f[i]/bassf) - np.round(12*np.log2(f[i]/bassf)))
        a[i] = a[i] * qFactor
    return [notes,t,a,f]
def computePCP(notes,a):
    #Given a block of our chromagram, find the 4 most prevalent notes and their "power" for the pitch classes
    cur = notes[0]
    new = np.zeros(12)
    for i in range(new.size):
        new[i] = np.sum(a[np.where(notes == note[i])])
    sorter = new.argsort()
    sort = new[sorter]
    bestFour = note[new.argsort()[new.size-1]]
    power = sort[new.size-1]
    for i in range(3):
            bestFour = np.append(bestFour,note[sorter[new.size-i-2]])
            power = np.append(power,sort[new.size-i-2])
    power = power/np.max(power)
    return [bestFour,power]
def computePCIT(x,blockSize,hopSize,fs = 44100):
    #Compile everything above together
    notes,t,a,f = computeChromagram(x,blockSize,hopSize,fs)
    pitchClasses = np.array([0,0,0,0])
    pitchClassPowers = np.array([0,0,0,0])
    for i in range(t.size):
        tempClass, tempPower = computePCP(notes,a[:,i])
        pitchClasses = np.vstack((pitchClasses,tempClass))
        pitchClassPowers = np.vstack((pitchClassPowers,tempPower))
    pitchClasses = pitchClasses[1:]
    pitchClassPowers = pitchClassPowers[1:]
    return [pitchClasses,pitchClassPowers,t]
