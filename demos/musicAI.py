from featureExtract.feature import calFeature
from classifier.model import MusicClassifier
from audioIO import record, load
import wave
import numpy as np
import matplotlib.pyplot as plt

# record the music
# filename = "../data/demo_chunks/exp.wav"
# frames, ex_samWid = record(filename, time = 10)
# feats, names = calFeature(filename)

# calculate params
feats, names = calFeature('../data/demo_chunks/dubstep.wav')

# load model
model = MusicClassifier("../data/model/dnn_3.h5")
model.getDataInfo("../data/data_set/beatsdataset.csv")

# predict
output = model.predict(feats)