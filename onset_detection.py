# onset detection file

import scipy as sci
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import math

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

def set_threshold(ac, pts=10):
    ac=np.sort(ac)[::-1]
    set_thresh=ac[pts-1]
    return set_thresh

def get_onsets(x, fs, hop_length = 200, frame_length = 2048):
    time = np.arange(0, source.size/fs,1/fs) # creates a time array
    energy = energy_fun(source, frame_length, hop_length, 1) #gets dB energy of audio
    rmse = rmse_fun(source, frame_length, hop_length, 1) #gets RMS dB energy of audio
    d_energy1=derivative(energy1)
    d_rmse1=derivative(rmse1)
    # onset detection
    onset_thresh=.5

    (detection_1, truth_detection_1)=onset_detect(d_rmse1, onset_thresh)
    (detection_2, truth_detection_2)=onset_detect(d_energy1, onset_thresh)

        
