# onset detection file

import scipy as sci
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import math
import matplotlib.pyplot as plt


def energy_fun(x, frame_length=1024, hop_length=512, log=0):
    if log==0:
        e = np.array([
        sum(abs(x[i:i+frame_length]**2))
        for i in range(0, len(x), hop_length)])
    else:
        try:
            e = np.array([ 
            20*math.log(sum(abs(x[i:i+frame_length]**2)), 10) # still getting issues
            for i in range(0, len(x), hop_length)])
        except:
            e = np.array([
            sum(abs(x[i:i+frame_length]**2))
            for i in range(0, len(x), hop_length)])
            for z in range(len(e)):
                if e[z]!= 0:
                    e[z]=20*math.log(e[z], 10)

    # converts energy from magnitude to dB - 20*math.log(sum(abs(x[i:i+frame_length]**2)), 10)
    # og version without the dB conversion sum(abs(x[i:i+frame_length]**2))
    return abs(e / np.max(e))

    pitchClasses = pitchClasses[1:]
    pitchClassPowers = pitchClassPowers[1:]
    return [pitchClasses,pitchClassPowers,t]

def rmse_fun(x, frame_length=1024, hop_length=512, log =0):
    e = energy_fun(x, frame_length, hop_length, log)
    e = np.sqrt(e / len(e))
    return e / np.max(e)

def derivative(x,fs_sig):
    # used to compute novelty functions
    d = x[1:] - x[:len(x)-1] # derivative/novelty
    # smooth the derivative function
    crit_freq=fs_sig/50
    filt =sig.butter(10, crit_freq, fs=fs_sig, output='sos') # low pass filter for smoothing set to fs/4 
    d = sig.sosfilt(filt, d) # applying the filter
    d[np.where(d<0)]=0 #HWR
    d=d / np.max(d) # normalizes data
    return d

def onset_detect(sound, thresh_pts=100):

    # old peak detection algorithm, leaving here just in case
    #detect= [x if x>=thresh else 0 for x in sound] # finds each audio sample with a magnitude above a threshold
    #detect[np.where(detect!=0)[0]]=0 # throws out the first peak 
    #truth_detect= detect>=thresh # boolean matrix of detect matrix

    # new peak detection 
    peak_locs=sig.find_peaks(sound,height=0)[0][1:]
    detect=np.zeros(sound.shape)
    detect[peak_locs]=sound[peak_locs]
    # finding the x highest peak value for thresholding
    ac=np.sort(detect)[::-1]
    set_thresh=ac[thresh_pts-1] 

    detect[np.where(detect<set_thresh)]=0
    detect=detect/np.ndarray.max(detect)
    peak_locs=sig.find_peaks(detect,height=0)[0][1:]
    #plt.plot(sound)
    #plt.plot(peak_locs, sound[peak_locs], "x")
    #plt.xlabel("block (hop length")
    #plt.ylabel("Amplitude of Smoothed Novelty Function (RMS dB)")
    #plt.title("Novelty function with detected peaks highlighted")
    #plt.show()
    return detect, peak_locs

#def set_threshold(ac, pts=10): # gives option to pick the N largest onsets instead of setting with a threshold
 #   ac=np.sort(ac)[::-1]
  #  set_thresh=ac[pts-1]
   # print(set_thresh)
    #return set_thresh

def get_onsets(x, fs, hop_length = 256, frame_length = 512, const_block=1):

    # this function takes in the raw audio file and either determines an optimal window length depending on the spacing between all of the onsets
    # or returns window  lengths for each window for chord detection based on spacing between all of the onsets
        # constant window length done, varying window length in progress

    # issue: silence at the begninning/end of a song  is thorowing this as well as chromogram function
    song_inds=np.where(x!=0)
    x=x[song_inds[0][0]:song_inds[0][-1]] # strips zeros at the beginning and end of the audio data

    time = np.arange(0, x.size/fs,1/fs) # creates a time array
    energy = energy_fun(x, frame_length, hop_length, 1) #gets dB energy of audio
    rmse = rmse_fun(x, frame_length, hop_length, 1) #gets RMS dB energy of audio

    #d_energy=derivative(energy,fs) # smoothed, HWR'ed novelty function for energy, have the energy in here as an alternative feature to look at for onsets
    d_rmse=derivative(rmse,fs) # smoothed, HWR'ed novelty function for rms energy
    # onset detection
    #onset_thresh=set_threshold(d_rmse, 5000) # picks the x number of the highest peaks to use for calculating a block length

    (detection_1, onset_indices_1)=onset_detect(d_rmse, 200)
    

    lens=np.where(detection_1>0)[0] # shows indices where theres a peak
    lengths = []
    for x in range(len(lens)):
        if x==0:
            lengths.append(lens[0])
        else:
            lengths.append(lens[x]-lens[x-1])
    if const_block: # finding the most frequent distances between onsets
        block_lens={}
        for x in range(len(lengths)):
            block=lengths[x]
            if block in block_lens.keys():
                block_lens[block]+=1
            else:
                block_lens[block]=1
        sorted_lens=sorted(block_lens.items(), key = lambda x:-x[1])
        if(sorted_lens[0][1]==sorted_lens[1][1]): #issue
            if (sorted_lens[0][0]>sorted_lens[1][0]):
                return (sorted_lens[0][0]*hop_length, onset_indices_1*hop_length) # returns the average distance between peaks in samples and the location of each detected peak 
            else:
                return (sorted_lens[1][0]*hop_length, onset_indices_1*hop_length) # returns the average distance between peaks in samples and the location of each detected peak 
        else:
            return (sorted_lens[0][0]*hop_length, onset_indices_1*hop_length) # returns the average distance between peaks in samples and the location of each detected peak 
