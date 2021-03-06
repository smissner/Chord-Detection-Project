

import scipy as sci
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as edist
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
npnote = np.array([
         "A",
         "Bb",
         "B",
         "C",
         "Db",
         "D",
         "Eb",
         "E",
         "F",
         "Gb",
         "G",
         "Ab"
    ])
freqs = dict((v, k) for k, v in note.items())

def noteName(f,bass,name):
    #This function will get the name of a note given a frequency as well as information pertaining to
    #what our bass note frequency is
    nAwayFromA = np.mod(np.round(12*np.log2(f/bass)),12)
    newnote = {}
    for i in range(12):
        newnote[i] = note.get(np.mod(freqs.get(name)+i,12))
    return (newnote.get(nAwayFromA,"zDC"))

def findTuning(f,a):
    muse = np.array([])
    tunedpoints = np.array([440.0])
    #Create an array of tuned notes on the equal tempered scale. This brute force method is suprisingly fast!
    while tunedpoints[tunedpoints.size-1] > 1:
        tunedpoints = np.concatenate((tunedpoints,[tunedpoints[tunedpoints.size-1]/1.059463]))
    while tunedpoints[tunedpoints.size-1] < 9000:
        tunedpoints = np.concatenate((tunedpoints,[tunedpoints[tunedpoints.size-1]*1.059463]))
    for i in range(a.shape[1]):
        #Given high points, calculate the distance from a tuned not and average them for the tuning frequency
        if np.any(a[:,i]>.2):
            for j in range(np.asarray(np.where(a[:,i]>.2)).size):
                newind = (np.abs(f - f[np.where(a[:,i]>.2)[0][j]]*2)).argmin()
                if a[newind,i] > .1:
                    muse = np.concatenate((muse,[f[np.where(a[:,i]>.1)[0][j]]]))
    diff = np.zeros(muse.size)
    for j in range(muse.size):
        closeid = (np.abs(tunedpoints - muse[j])).argmin()
        diff[j] = muse[j] - tunedpoints[closeid]
    return 440 * 2**(np.sum(diff)/1200)

def computeChromagram(x,blockSize,hopSize,fs):
    #Take the spectrogram of our audio data, and normalize it
    f,t,a = sig.spectrogram(x,fs,nperseg = blockSize,noverlap = blockSize - hopSize)
    a = a/np.max(a)
    #Apply filtering envelope. Hard cutoff lowpass, log based rolloff on highpass
    a[np.where(f<80)[0]] = a[np.where(f<80)[0]] * 0
    """
    env = np.logspace(1,0,np.where(f>500)[0].size)
    for i in range(t.size):
        a[np.where(f>500)][:,i] = a[np.where(f>500)][:,i] * env
    """
    tunef = findTuning(f,a)
    tunenote = "A"
    notes = np.array([])
    newa = np.zeros([12,a.shape[1]])
    #Create a new note array to replace the frequency array so that we have a chromatic representation of the spectrogram
    amps = 0
    n = 1
    pretempnote = "zDC"
    #Get note bins from frequency bins
    for i in range(f.size):


        qFactor = 1 - abs(12*np.log2(f[i]/tunef) - np.round(12*np.log2(f[i]/tunef)))
        a[i] = a[i] * qFactor



        #This qFactor will hopefully work to make out of tune notes not mess with our data(say someone in the recording plays
        #a really sharp c, we don't want that being recorded as a Db). Works by reducing the value of notes in frequency bins
        #far from a centralized note

        tempnote = noteName(f[i],tunef,tunenote)
        notes = np.append(notes,tempnote)
        if i>0 and notes[i] == notes[i-1]:
            amps = amps + a[i]
            n = n + 1
        else:
            newa[freqs.get(pretempnote),:] = newa[freqs.get(pretempnote),:] + amps/n
            amps = 0
            n = 1
        pretempnote = tempnote


    return [notes,t,newa,f]
def computePCP(notes,a):
    #Given a block of our chromagram, find the 4 most prevalent notes and their "power" for the pitch classes
    power = np.zeros(12)
    for i in range(power.size):
        power[i] = np.sum(a[i])
    power = power/np.max(power)
    return [power]

def computePCIT(x,blockSize,hopSize, peak_array, fs = 44100):
    # time stamps is an array input that contains every point where an onset was detected
    #Compile everything above together
    notes,t,a,f = computeChromagram(x,blockSize,hopSize,fs)
    pitchClassPowers = np.zeros(12)
    for i in range(t.size):
        tempPower = computePCP(notes,a[:,i])
        pitchClassPowers = np.vstack((pitchClassPowers,tempPower))
    pitchClassPowers = pitchClassPowers[1:]

    # plotting chromagram for visualization
    #plotting_chroma = [[pitchClassPowers[j][i] for j in range(len(pitchClassPowers))] for i in range(len(pitchClassPowers[0]))]
    #plt.imshow(plotting_chroma, cmap='gray_r', aspect='auto', origin='lower')
    #plt.ylabel("notes")
    #plt.yticks(np.arange(12), 'A Bb B C Db D Eb E F Gb G Ab'.split())
    #plt.xlabel("time")
    #plt.title("Chromagram with constant blocks")
    #plt.show()
    # averaging chromagram estimates between peaks
    time_peaks=peak_array/fs # converts peak indices into time indices
    block_peaks=peak_array/hopSize # converts peak indices into hop indices corresponding to the computed block size
    avgd_chroma_blocks=np.zeros((len(block_peaks), 12))
    for z in range(len(block_peaks)):
        if z==len(block_peaks):
            avgd_chroma_blocks[z,:]=np.average(pitchClassPowers[round(block_peaks[z]-1):], axis=0)
        elif z==0:
            avgd_chroma_blocks[z,:]=np.average(pitchClassPowers[0:round(block_peaks[z])], axis=0)
        else:
            avgd_chroma_blocks[z,:]=np.average(pitchClassPowers[round(block_peaks[z]-1):round(block_peaks[z])], axis=0)
    #plotting_chroma_avg = [[avgd_chroma_blocks[j][i] for j in range(len(avgd_chroma_blocks))] for i in range(len(avgd_chroma_blocks[0]))]
    #plt.imshow(plotting_chroma_avg, cmap='gray_r', aspect='auto', origin='lower')
    #plt.ylabel("notes")
    #plt.yticks(np.arange(12), 'A Bb B C Db D Eb E F Gb G Ab'.split())
    #plt.xlabel("time")
    ##plt.xticks(time_peaks)
    #plt.title("Chromagram with averaged blocks")
    #plt.show()
    return [avgd_chroma_blocks,time_peaks]

def correlateChords(notes, flag_7):
    #print(notes)
    # does cross correlation between the chromagram and masks for chords
    if flag_7: # if we want to try 7th chords
        chord_masks=[
            #[1,0,0,0,0,0,0,1,0,0,0,0], # power chord (just root--5)
            [1/3,0,0,0,1/3,0,0,1/3,0,0,0,0], # maj
            [1/3,0,0,1/3,0,0,0,1/3,0,0,0,0], # min
            [1/3,0,0,0,1/3,0,0,0,1/3,0,0,0], # aug
            [1/3,0,0,1/3,0,0,1/3,0,0,0,0,0], # dim
            [1/4,0,0,0,1/4,0,0,1/4,0,0,0,1/4], # maj7
            [1/4,0,0,0,1/4,0,0,1/4,0,0,1/4,0], # dom7
            [1/4,0,0,1/4,0,0,0,1/4,0,0,1/4,0], # min7
            [1/4,0,0,0,1/4,0,0,0,1/4,0,1/4,0], # aug7
            [1/4,0,0,1/4,0,0,0,1/4,0,0,0,1/4], # minmaj7
            [1/4,0,0,1/4,0,0,1/4,0,0,0,1/4,0], # halfdim7
            [1/4,0,0,1/4,0,0,1/4,0,0,1/4,0,0], # dim7
            ]
    else: # just triads
        chord_masks=[
            #[1,0,0,0,0,0,0,1,0,0,0,0], # power chord (just root--5)
            [1/3,0,0,0,1/3,0,0,1/3,0,0,0,0], # maj
            [1/3,0,0,1/3,0,0,0,1/3,0,0,0,0], # min
            [1/3,0,0,0,1/3,0,0,0,1/3,0,0,0], # aug
            [1/3,0,0,1/3,0,0,1/3,0,0,0,0,0] # dim
            ]

    # arrays of chord and key names for use with the euclidean distance matrix later
    chords=np.array(["maj","min","aug","dim","maj7","7","min7","aug7","minmaj7","halfdim7","dim7"]) # rows

    corrs=np.ones([len(chord_masks), 12])

    for y in range(len(chord_masks)):
        mask=np.array(chord_masks[y])
        for x in range(12):
            #corrs[y,x]=edist.euclidean(np.array(notes,dtype='int64'), np.array(np.roll(mask, x),dtype='int64')) # maybe replace with sum(corr) if euclidean doesnt work
            #corrs[y,x]=np.correlate(np.array(notes,dtype='int64'), np.array(np.roll(mask, x),dtype='int64')) # trying numpy correlation function- getting weird results

            # writing my own correlation function because I am kinda frustrated
            autocorr_sig=mask[:12-x]
            corrs[y,x]=np.dot(notes[x:], autocorr_sig)

    # should now have a matrix of euclidean distances, indices correspond to the key(column) and chord type (row)
        # need to find indices of the minimun value in the matrix
    best_dist=np.amax(abs(corrs))
    result = np.where(corrs == best_dist) # finds the indices for the best matching chord

    if len(result[0])==1:
        #print(result)
        best_chord=npnote[result[1][0]]+":"+chords[result[0][0]] # making the chord name
    else:
        best_chord="N"
    print(best_chord)

    return best_chord
    #return (best_chord, best_dist) # return the name of the detected chord along with the euclidean distance for the chord just in case




def findchordnotes(pitchClassPowers, pitchcount):
    indexes = np.array(np.argpartition(pitchClassPowers, -pitchcount)[-pitchcount:])
    return (npnote[indexes], pitchClassPowers[indexes])
