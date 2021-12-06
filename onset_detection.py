# onset detection file

import scipy as sci
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import math


######## IDEAS FOR IMPROVEMENT:
        # Use filterbanks, filter certain freqs out
            # maybe try lpf first since bass normal;y keeps tempo in pop (bass, kick)
        # use some sort of novelty function based on a difference measure in energy with HWR



def energy_fun(x, frame_length=1024, hop_length=512, log=0):
    if log==0:
        e = np.array([
        sum(abs(x[i:i+frame_length]**2))
        for i in range(0, len(x), hop_length)])
    else:
        e = np.array([
        20*math.log(sum(abs(x[i:i+frame_length]**2)), 10)
        for i in range(0, len(x), hop_length)])
    
    # converts energy from magnitude to dB - 20*math.log(sum(abs(x[i:i+frame_length]**2)), 10)
    # og version without the dB conversion sum(abs(x[i:i+frame_length]**2))
    return abs(e / np.max(e))

    pitchClasses = pitchClasses[1:]
    pitchClassPowers = pitchClassPowers[1:]
    return [pitchClasses,pitchClassPowers,t]

def rmse_fun(x, frame_length=1024, hop_length=512, log =0):
    e = energy_fun(x, frame_length, hop_length)
    e = np.sqrt(e / len(e))
    return e / np.max(e)

def derivative(x):
    d = np.abs(x[1:] - x[:len(x)-1])
    return d / np.max(d)

def onset_detect(sound, thresh=.8):
    detect= [x if x>=thresh else 0 for x in sound] # finds each audio sample with a magnitude above a threshold
    #truth_detect= detect>=thresh # boolean matrix of detect matrix
    detect=np.array(detect)
    
    truth_detect= [True if x else False for x in sound]
    return detect, truth_detect

def set_threshold(ac, pts=10): # gives option to pick the N largest onsets instead of setting with a threshold
    ac=np.sort(ac)[::-1]
    set_thresh=ac[pts-1]
    return set_thresh

def get_onsets(x, fs, hop_length = 256, frame_length = 512, onset_thresh=.75):

    # this function takes in the raw audio file and either determines an optimal window length depending on the spacing between all of the onsets 
        # constant window length done

    # leave const_block on for now, 0 is for if we can figure out how to do non constant block lengths

    time = np.arange(0, x.size/fs,1/fs) # creates a time array
    energy = energy_fun(x, frame_length, hop_length, 1) #gets dB energy of audio
    rmse = rmse_fun(x, frame_length, hop_length, 1) #gets RMS dB energy of audio
    d_energy=derivative(energy)
    d_rmse=derivative(rmse)
    # onset detection

    (detection_1, truth_detection_1)=onset_detect(d_rmse, onset_thresh)

    lens=np.where(detection_1>=onset_thresh)[0] # shows indices where theres a peak
    lengths=[]
    for x in range(len(lens)):
        if x==0:
            lengths[0]==lens[0]
        else:
            lengths[x]=lens[x]-lens[x-1]

    block_lens={}
    for x in range(len(lengths)):
        block=lengths[x]
        if block in block_lens.keys():
            block_lens[block]+=1
        else:
            block_lens[block]=1
    sorted_lens=sorted(block_lens.items(), key = lambda x:-x[1]) # sorts by the 

    if(sorted_lens[0][1]==sorted_lens[1][1]):
        if (sorted_lens[0][0]>sorted_lens[1][0]):
            return sorted_lens[0][0]*hop_length
        else:
            return sorted_lens[1][0]*hop_length
    else:
        return sorted_lens[0][0]*hop_length









        
