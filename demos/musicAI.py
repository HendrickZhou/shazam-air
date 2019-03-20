from featureExtract.feature import calFeature
from classifier.model import MusicClassifier
from audioIO import record, load
import wave
import numpy as np
import matplotlib.pyplot as plt

# record the music
# frames, ex_samWid = record("./data/demo_chunks/exp.wav", time = 10)
# wav, f = load("./data/demo_chunks/exp.wav", sr = 22050)

# calculate params
feats, names = calFeature('./data/demo_chunks/dubstep.wav')

# load model
model = MusicClassifier("./data/model/dnn_3.h5")
model.getDataInfo("./data/data_set/beatsdataset.csv")

# predict
output = model.predict(feats)
print(output)