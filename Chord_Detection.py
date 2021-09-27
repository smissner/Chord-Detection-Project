
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
    
    