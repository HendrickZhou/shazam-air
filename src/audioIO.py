import numpy as np
import librosa
from librosa import display
import pyaudio as pa
import wave
import scipy.io.wavfile as sciwavf
import matplotlib.pyplot as plt
import struct      

"""
Basic wav file i/o
"""
def record(filename, time = 5, rate = 22050, chunk = 1024):
    CHUNK = chunk
    FORMAT = pa.paInt16
    CHANNELS = 1
    RATE = rate
    RECORD_SECONDS = time
    WAVE_OUTPUT_FILENAME = filename

    p = pa.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []
    
        
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        
        
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))    
    wf.close()

def play(filename):
    CHUNK = 1024
    
    wf = wave.open(filename, 'rb')

    p = pa.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data != b'':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()

def store(filename, chunk, f):
    sciwavf.write(filename, f, chunk)

def load(filename, sr = 22050):
    wav, f = librosa.load(filename, sr = 22050)
    print('wav file infomation:')
    print('length of wav file: %d' % len(wav))
    print('sampling frequency is: %d' % f)
    return wav, f

"""
Raw data operation
"""
def getRawData(filename):
    CHUNK = 1024
    wf = wave.open(filename, 'rb')
    p = pa.PyAudio()
    
    raw_data = []

    data = wf.readframes(CHUNK)

    while data != b'':
        # stream.write(data)
        data = wf.readframes(CHUNK)
        raw_data.append(data)

    p.terminate()
    return raw_data

def decodePCM(rawData):  
    npts=len(rawData)                     
    formatstr='%dh' % (npts/2)                
    int_data=struct.unpack(formatstr,rawData) 
    f_data = np.array([float(val) / pow(2, 15) for val in int_data])
    return f_data









