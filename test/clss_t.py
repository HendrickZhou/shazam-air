from classifier.prepareData import prepareData

dataColRanges = [('1-ZCRm', '34-ChromaDeviationm'), ('69-BPM', '71-BPMessentia')]
labelCol = 'class'
a = Classifer_dataset("../data/data_set/beatsdataset.csv")
a.perpareDatax(dataColRanges, labelCol)