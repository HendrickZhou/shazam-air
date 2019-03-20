from featureExtract.feature import calFeature
import classifier.model.MusicClassifier as ai 
from audioIO import record, load
import wave
import numpy as np
import matplotlib.pyplot as plt

# record the music
frames, ex_samWid = record("./exp.wav", time = 2)
wav, f = load("./exp.wav", sr = 22050)

# calculate params
feats, names = calFeature('./dubstep.wav')

# load model
model = ai("./example.h5")
model.getDataInfo("../data/data_set/beatsdataset.csv")

# predict
output = model.predict(feats)