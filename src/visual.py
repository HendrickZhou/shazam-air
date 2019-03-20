import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
from scipy.signal import stft
import numpy as np




"""
Plotting wav data
"""
def plotWav(wav, f, ratio, start, end, title, mode='t'):
    width, height = plt.figaspect(ratio)
    fig = plt.figure(figsize=(width,height))
    if mode == 't': #time
        startI = start * f
        endI = end * f
        Idx = np.arange(startI, endI)
        chunk = wav[startI : endI]
        plt.plot(Idx, chunk)
        plt.title(title)
        return chunk
    elif mode == 'i': # index
#         startSec = start / f
#         endSec = end / f
#         secIdx = np.arange(startSec, endSec, 1/f)
        Idx = np.arange(start, end)
        chunk = wav[start : end]
        plt.plot(Idx, chunk)
        plt.show()
        plt.title(title)
        return fig



# def plotSpectrum():
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(wav[274520:1543504])), ref=np.max)
#     width, height = figaspect(0.1)
#     fig1 = figure(figsize=(width,height))
#     librosa.display.specshow(D, y_axis='linear')
#     plt.title("lossless")

def plotSpect(specdb, ratio, title):
    width, height = figaspect(ratio)
    fig = figure(figsize=(width,height))
    
    librosa.display.specshow(specdb, sr = 22050, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title(title)


def fft_bar_data(data_seq, bar_number):
    """
    suggest param range:
        bar_number: ~100
        half: 500~1000
     
    one solution: 0-padding freq to fixed length like 1000
    let's try 1000 first
    
    bar_size has granrantee the safety for this function
    """
    # to do: add robust safety check
    zero_padding = 1000 # by default
    data_freq = abs(fftshift(fft(data_seq, n=zero_padding))) # zero-padding to 2000
    bar_len = int((zero_padding/2) / bar_number) # THIS IS WTFFFF!!!!
    
    data_bar = np.zeros(bar_number)
    for i in range(bar_number):
        data_bar[i] = np.mean(data_freq[bar_len*i : bar_len*(i + 1)])
        
    return data_bar