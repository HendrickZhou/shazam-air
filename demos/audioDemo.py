import threading
import math
from queue import Queue
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyaudio as pa
import wave
from scipy import signal, fftpack
from audioIO import decodePCM, load
from visual import fft_bar_data, plotSpect, plotWav


def consum(out_q, bar_no_idx):
    bar_size = [10, 50, 100, 200, 250]
    bar_number = bar_size[bar_no_idx]

    freq_idx = np.r_[:bar_number]                                    
    freq_magn = np.zeros(bar_number)
    
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_autoscaley_on(False)
    ax.set_ylim([0, 20])
    while 1:
        raw_data = out_q.get(timeout = 0.3)
        
        data_bar = fft_bar_data(decodePCM(raw_data), bar_number)
        data_bar = data_bar
        ax.cla()
        ax.bar(freq_idx, data_bar, width=0.3)

        fig.canvas.draw()
        fig.canvas.flush_events()
        
def produce(in_q, filename):
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
        in_q.put(data)
        
    stream.stop_stream()
    stream.close()

    p.terminate()
    
def realTime(filename):
    q = Queue()
    
    t1 = threading.Thread(target=produce, args=(q, filename)) 
    t2 = threading.Thread(target=consum, args=(q, 0)) 
        
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()

    

"""
Basic plotting
"""
wav_rec, f_rec = load("./data/demo_chunks/rec.wav", sr = 22050)
wav_raw, f_raw = load("./data/demo_chunks/raw.wav", sr = 22050)

plotWav(wav_rec, f_rec, 0.1, 0, 10, "rec chunk")
plotWav(wav_raw, f_raw, 0.1, 0, 10, "raw chunk")

plotSpect(wav_rec, 0.2, "recording sepctrum")
plotSpect(wav_raw, 0.2, "raw data spectrum")


"""
Real-time spectrum demo
"""
print("plotting the real-time spectrum")
# demo_file = "./data/demo_chunks/real_time_demo.wav"
# realTime(demo_file)
