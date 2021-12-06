import scipy as sci
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
#Instantiate static variables
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
def noteName(f,bass,name):
    #This function will get the name of a note given a frequency as well as information pertaining to
    #what our bass note frequency is
    nAwayFromA = np.mod(np.round(12*np.log2(f/bass)),12)
    newnote = {}
    for i in range(12):
        newnote[i] = note.get(np.mod(freqs.get(name)+i,12))
    return (newnote.get(nAwayFromA,"zDC"))
def get_spectral_peaks(a,numPeaks):
    a = a**2
    peaks = np.zeros(numPeaks)
    for i in range(a.shape[1]):
        currentPeaks = sig.find_peaks(a[:,i])
        mask = np.zeros(a.shape[0])
        mask[currentPeaks[0]] = 1
        a[:,i] = a[:,i] * mask
    maxpeakf = np.zeros(a.shape[0])
    for j in range(a.shape[0]):
        maxpeakf[j] = np.max(a[j,:])
    for k in range(numPeaks):
        peaks[k] = np.argmax(maxpeakf)
        maxpeakf[np.argmax(maxpeakf)] = 0
    return np.asarray(peaks,dtype = int)



def findTuning(f,a):
    peaks = get_spectral_peaks(a,50)
    tunedpoints = np.array([440.0])
    #May be a numpy way to do this hopefully cause this isn't very efficient    
    while tunedpoints[tunedpoints.size-1] > 1:
        tunedpoints = np.concatenate((tunedpoints,[tunedpoints[tunedpoints.size-1]/1.059463]))
    while tunedpoints[tunedpoints.size-1] < 9000:
        tunedpoints = np.concatenate((tunedpoints,[tunedpoints[tunedpoints.size-1]*1.059463]))
    diff = np.zeros(peaks.size)
    for j in range(peaks.size):
        closeid = (np.abs(tunedpoints - f[peaks[j]])).argmin()
        diff[j] = 1200*np.log2(f[peaks[j]]/tunedpoints[closeid])
    return 440 * 2**(diff[np.argmax(abs(diff))]/1200)
def computeChromagram(x,blockSize,hopSize,fs):
    #Take the spectrogram of our audio data, and normalize it
    f,t,a = sig.spectrogram(x,fs,nperseg = blockSize,noverlap = blockSize - hopSize)
    a = a/np.max(a)
    #Create a tuning bassline given the lowest prevalent note in the audio. This process seems to be helpful, but only very
    #slightly I've found, and due to possible complications this process creates, we may want to remove it.
    a[np.where(f<80)[0]] = a[np.where(f<80)[0]] * 0
    for i in range(np.where(f>500)[0].size):
        a[np.where(f>500)[0][i]] = a[np.where(f>500)[0][i]] * 1/(np.log10(i)+1)
    tunef = findTuning(f,a)
    tunenote = "A"
    notes = np.array([])
    newa = np.zeros([12,a.shape[1]])
    #Create a new note array to replace the frequency array so that we have a chromatic representation of the spectrogram
    amps = 0
    n = 1
    pretempnote = "zDC"
    for i in range(f.size):
        tempnote = noteName(f[i],tunef,tunenote)
        notes = np.append(notes,tempnote)
        qFactor = 1 - abs(12*np.log2(f[i]/tunef) - np.round(12*np.log2(f[i]/tunef)))
        a[i] = a[i] * qFactor

            

        #This qFactor will hopefully work to make out of tune notes not mess with our data(say someone in the recording plays
        #a really sharp c, we don't want that being recorded as a Db). Works by reducing the value of notes in frequency bins
        #far from a centralized note
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
def computePCIT(x,blockSize,hopSize,fs = 44100):
    #Compile everything above together
    notes,t,a,f = computeChromagram(x,blockSize,hopSize,fs)
    pitchClassPowers = np.zeros(12)
    for i in range(t.size):
        tempPower = computePCP(notes,a[:,i])
        pitchClassPowers = np.vstack((pitchClassPowers,tempPower))
    pitchClassPowers = pitchClassPowers[1:]
    return [pitchClassPowers,t]


def findchordnotes(pitchClassPowers, pitchcount):
    indexes = np.array(np.argpartition(pitchClassPowers, -pitchcount)[-pitchcount:])
    return (npnote[indexes], pitchClassPowers[indexes])
