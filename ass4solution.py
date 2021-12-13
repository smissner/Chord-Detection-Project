import numpy as np
from scipy.io import wavfile
import math
import scipy as sp
import os
import io
from scipy.spatial import distance as edist
#import matplotlib.plt as plt

KEYAUDIO = './key_tf/key_eval/audio/'
KEYGT = './key_tf/key_eval/GT/'
TFAUDIO = './key_tf/tuning_eval/audio/'
TFGT = './key_tf/tuning_eval/GT/'



"""

LERCH NOTES:

DONE! i think.
[-2.5] get_spectral_peaks: can not handle the case of less than 20 peaks,
but more importantly: a high value is not a peak!

DONE!
[-1] estimate_tuning_freq: your bin to freq equation is off,
replace with
spectral_peaks_hz = np.array(spectral_peak_columns) * fs / 2.0 / (np.shape(spectogram)[0]-1)


FUCK
[-2] estimate_tuning_freq: it's fine to use np.histogram, however, you have to use it correctly.
First, you do not want the histogram bins to change based on the audio input,
so you want to predefine them. Second, hist[1] contains the *bin edges*,
not the bin centers!


"""
"""


A. Tuning Frequency Estimation: [25points]


Helper Functions

"""

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

#reference solution for compute_spectogram audio ->
def compute_spectrogram(xb):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])

    for n in range(0, numBlocks):
        # apply window
        tmp = abs(np.fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]

        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))]
        X[[0, math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2)
        #let's be pedantic about normalization
    return X
#reference solution for block audio ->
def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)

#reference solution for freq to cents/midi ->

def convert_freq2cents(fInHz, fA4InHz = 440):
    def convert_freq2cents_scalar(f, fA4InHz):
        if f <= 0:
            return 0
        else:
            return 100*(69 + 12 * np.log2(f/fA4InHz))
    fInHz = np.asarray(fInHz)

    if fInHz.ndim == 0:
       return convert_freq2cents_scalar(fInHz,fA4InHz)

    if fInHz.ndim == 1:
        midi = np.zeros(fInHz.shape)
        for k, f in enumerate(fInHz):
            midi[k] =  convert_freq2cents_scalar(f, fA4InHz)
        return (midi)
    if fInHz.ndim == 2:
        midi = np.zeros(fInHz.shape)
        for k, f in enumerate(fInHz):
            for j, f2 in enumerate(f):
                midi[k,j] =  convert_freq2cents_scalar(f2, fA4InHz)
        return (midi)

"""
A. Tuning Frequency Estimation: [25points]

1. [5 points] Write a function [spectralPeaks] = get_spectral_peaks(X) that
returns the top 20 spectral peak bins of each column of magnitude spectrogram X.
"""
def get_spectral_peaks(X):
    #return np.array(np.argpartition(X, -20)[-20:])
    peaks = np.zeros((X.shape[1],20))
    for i in range (X.shape[1]):
        #old
        #peaks[i] = np.array(np.argpartition(X[:,i], -20)[-20:])
        peaks[i] = np.array(np.argwhere(X[:,i] == 1.0))
    return peaks

"""
2. [20 points] Write a function [tfInHz] = estimate_tuning_freq(x, blockSize, hopSize, fs)
to return the tuning frequency. x is the time domain audio signal, blockSize is
the block size, hopSize is the hopSize, and fs is the sample rate.
Use the deviation from the equally tempered scale in Cent for your estimate.
You will use get_spectral_peaks() function to obtain the top 20 spectral peaks
for each block. Use functions from the reference solutions for previous assignments
 for blocking and computing the spectrogram. For each block, compute the deviation
 of the peak bins from the nearest equally tempered pitch in cents.
 Then, compute a histogram of the deviations in Cent and derive the tuning frequency
 from the location of the max of this histogram.
 Make sure your function works for any blockSizes.
"""

def estimate_tuning_freq(x, blockSize, hopSize, fs):
    (xb, t) = block_audio(x, blockSize, hopSize, fs)

    spectogram = compute_spectrogram(xb)
    spectral_peak_columns = get_spectral_peaks(spectogram)
    spectral_peaks_hz = np.array(spectral_peak_columns) * fs / 2.0 / (np.shape(spectogram)[0]-1)

    spectral_peaks_cents = convert_freq2cents(spectral_peaks_hz)
    nearest_cents = np.round(spectral_peaks_cents, -2)
    deviation =np.subtract(spectral_peaks_cents.flatten(), nearest_cents.flatten())
    hist = np.histogram(deviation)
    print(f"histogram is {hist}")
    cents_dev = hist[1][np.argmax(hist[0])]

    def find_tuning(cent, fA4InHz = 440):
        return fA4InHz * np.power(2, (cent / 100.0) / 12.0)
    val = find_tuning(cents_dev)
    return val


# def test_estimate_tuning_freq():
#     path = './key_tf/tuning_eval/audio/cycling_road.wav'
#     (fs, x) = wavfile.read(path)
#     val = estimate_tuning_freq(x, 4096, 2048, fs)

#     print(val)


"""
B. Key Detection: [50 points]
 f
1. [25 points] Write a function [pitchChroma] = extract_pitch_chroma(X, fs, tfInHz)
which returns the pitch chroma array (dimensions 12 x numBlocks).
X is the magnitude spectrogram, fs is the sample rate, and tfInHz is the tuning frequency.
Compute the pitch chroma for the 3 octave range C3 to B5.
You will need to adjust the semitone bands based on your calculation of tuning frequency
deviation. Each individual chroma vector should be normalized to a length of 1.
"""
def extract_pitch_chroma(X, fs, tfInHz):

    octaves = 3
    H = np.zeros([12,X.shape[0]])
    output = np.zeros([12,X.shape[1]])
    boundries= [tfInHz * np.power(2, i/12.0) for i in np.arange(-21,15)]
    boundries2 = np.zeros((12,octaves))
    for i in range(octaves):
        boundries2[:,i] = boundries[i*12:i*12+12]
    for i in range(boundries2.shape[0]):
        bound = np.zeros(2)
        for j in boundries2[i]:
            bound[0] = int(math.ceil(np.power(2,-1/(12)*2)*j*2*(X.shape[0]-1)/fs))
            bound[1] = int(math.ceil(np.power(2,1/(12)*2)*j*2*(X.shape[0]-1)/fs))
            try:
                H[i,int(bound[0]):int(bound[1])]= 1/ (bound[1]-bound[0])
            except ZeroDivisionError:
                H[i,int(bound[0]):int(bound[1])]= 1
    output = np.dot(H,np.power(X,2))
    norm = output.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1
    output = output / norm
    return output


"""

2. [25 points] Write a function [keyEstimate] = detect_key(x, blockSize, hopSize, fs, bTune)
to detect the key of a given audio signal.
The parameter bTune is True or False and will specify if tuning frequency correction
is done for computing the key. The template profiles to use for estimating the key are
the Krumhansl templates:


Note that this array contains both major and minor key profiles, respectively.
 Hint: don't forget to normalize the key profiles. Use Euclidean distance.

"""
# key names
#cKeyNames = np.array(['C Maj', 'C# Maj', 'D Maj', 'D# Maj', 'E Maj', 'F Maj', 'F# Maj', 'G Maj', 'G# Maj', 'A Maj', 'A# Maj', 'B Maj',
                     #'c min', 'c# min', 'd min', 'd# min', 'e min', 'f min', 'f# min', 'g min', 'g# min', 'a min', 'a# min', 'b min'])





def detect_key(x, blockSize, hopSize, fs, bTune):
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    t_pc = t_pc / t_pc.sum(axis=1, keepdims=True)
    key_names = np.array(['C Maj', 'C# Maj', 'D Maj', 'D# Maj', 'E Maj', 'F Maj', 'F# Maj', 'G Maj', 'G# Maj', 'A Maj', 'A# Maj', 'B Maj',
                         'c min', 'c# min', 'd min', 'd# min', 'e min', 'f min', 'f# min', 'g min', 'g# min', 'a min', 'a# min', 'b min'])

    t_pc[0] = np.roll(t_pc[0], -3)
    t_pc[1] = np.roll(t_pc[1], -3)

    if bTune:
        tfInHz = estimate_tuning_freq(x, blockSize, hopSize, fs)
    else:
        tfInHz = 440
    xb,tInSec = block_audio(x,blockSize,hopSize,fs)
    X = compute_spectrogram(xb)
    output = extract_pitch_chroma(X, fs, tfInHz)
    output_means = np.zeros(output.shape[0])
    for i in range(output.shape[0]):
        output_means[i] = np.mean(output[i,:])

    d_major = np.zeros(output.shape[0])
    d_minor = np.zeros(output.shape[0])
    for i in range(12):
        d_major[i] = edist.euclidean(output_means,np.roll(t_pc[0],i))
        d_minor[i] = edist.euclidean(output_means,np.roll(t_pc[1],i))
        # d_major[i] = np.sum(np.abs(output_means-np.roll(t_pc[0],i)))
        # d_minor[i] = np.sum(np.abs(output_means-np.roll(t_pc[1],i)))
    distance = np.concatenate((d_major,d_minor))
    key = np.argmin(distance)
    #Est_Key = key_names[key]
    return key


"""

C. Evaluation: [25 points]

For the evaluation use blockSize = 4096, hopSize = 2048.

1. [10 points] Write a function [avgDeviation] = eval_tfe(pathToAudio, pathToGT)
 that evaluates tuning frequency estimation for the audio files in the folder pathToAudio.
 For each file in the audio directory, there will be a corresponding .txt file
 in the GT directory containing the ground truth tuning frequency.
 You return the average absolute deviation of your tuning frequency estimation
 in cents for all the files.
"""
def eval_tfe(pathToAudio, pathToGT):
    searchPath = pathToAudio
    otherPath = pathToGT
    dev = []
    for root, dir, files in os.walk(searchPath):
        for f in files:
            print(f)
            (fs, x) = wavfile.read(f"{searchPath}{f}")
            res = estimate_tuning_freq(x, 4096, 2048, fs)
            answerPath =  f"{otherPath}{f.split('.')[0]}.txt"
            with open(answerPath) as ans:
                line = ans.readlines()
            correct = float(line[0])
            print(f"deviation calculations {res} - {correct}")
            dev.append(convert_freq2cents(res) - convert_freq2cents(correct))
    # print(f"all deviations {dev}")
    return np.mean(dev)

print("tuning freq mean deviation is: ", eval_tfe(TFAUDIO,  TFGT))
"""
2. [10 points] Write a function [accuracy] = eval_key_detection(pathToAudio, pathToGT)
that evaluates key detection for the audio files in pathToAudio.
For each file in the audio directory, there will be a corresponding .txt file in
the GT directory containing the ground truth key label.
You return the accuracy = (number of correct key detections) / (total number of songs)
of your key detection for all the files with and without tuning frequency estimation.
The output accuracy will be an np.array dimensions 2 x 1 vector with the first element
the accuracy with tuning frequency correction and the second without tuning frequency
correction.
"""
def eval_key_detection(pathToAudio, pathToGT):
    searchPath = pathToAudio
    otherPath = pathToGT
    dev = []
    dev_tuned = []
    for root, dir, files in os.walk(searchPath):
        for f in files:
            print(f)
            (fs, x) = wavfile.read(f"{searchPath}{f}")
            res = detect_key(x, 4096, 2048, fs, False)
            res_tuned = detect_key(x, 4096, 2048, fs, True)
            answerPath =  f"{otherPath}{f.split('.')[0]}.txt"
            with open(answerPath) as ans:
                line = ans.readlines()
            correct = float(line[0])
            print(f"key deviation calculations {res} - {correct}")
            dev.append(res - correct)
            # please remove the +3 when bTune is functional
            dev_tuned.append(res_tuned - correct)
        songs = len(dev)
        rightsongs = dev.count(0)
        acc = rightsongs / songs
        rightsongs_tuned = dev_tuned.count(0)
        acc_tuned = rightsongs_tuned / songs
        accuracy = np.array([acc, acc_tuned])
    print(f"accuracy {accuracy}")
    return (accuracy)

print("key estimation accuracy is:", eval_key_detection(KEYAUDIO, KEYGT))

"""
3. [5 points] Write a function [avg_accuracy, avg_deviationInCent] =
evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf)
which runs the above two functions with the data given  Download data givenin
the respective directories. Report the average absolute deviation for the tuning
 frequency estimation in Cent and the accuracy for key detection with and without
 tuning frequency correction.
"""

def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):
    dev = eval_tfe(pathToAudioTf, pathToGTTf)
    acc = eval_key_detection(pathToAudioKey, pathToGTKey)
    return (acc, dev)

if __name__ == "__main__":

    avg_accuracy, avg_deviationInCent = evaluate(KEYAUDIO, KEYGT, TFAUDIO, TFGT)
    print("The average absolute deviation for the tuning frequency estimation in Cent: ", avg_deviationInCent)
    print("The accuracy for key detection without tuning frequency correction: ", avg_accuracy[0])
    print("The accuracy for key detection with tuning frequency correction: ", avg_accuracy[1])
