import numpy as np
import pandas as pd
 
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

class Classifier_dataset:
    """
    This class only works for dataframe that fit the need of classifier
    

    attributes:
    ****************************
    _f : filename

    _frameData : raw data
    _data : dataframe
    _label : dataframe

    _nClass : number of label classes

    _dataArray : np array of data
    _labelArray : one-hot coded np array

    _shuff

    _labelMap : mapping from label decimal number to string
    ****************************

    methods:
    ****************************
    MAIN
        prepareData
    TOOLS
        divideLabelAndData
        extractLabelInfo
        generateMapping
        one_hot_key_encode
        one_hot_key_decode
    PLOTTING
        plotLDA
        plotPCA
    ****************************


    """

    def __init__(self, f):
        self._f = f
        try:

            self._frameData = pd.read_csv(f)
        except:
            print("fail to open csv")

    def prepareData(self, dataColRanges, labelCol, shuffle = False):
        """
        return label, data
        """
        if shuffle == True:
            self._frameData = self._frameData.sample(frac=1).reset_index(drop=True)


        self.divideLabelAndData(dataColRanges, labelCol)
        self.generateMapping()

        # Process label
        labelArray = []
        for i, value in self._label.iteritems():
            labelArray.append(self.one_hot_key_encode(value))
        self._labelArray = np.array(labelArray)

        # Process data
        # for pandas version 0.24.2 use this line instead
        # self._dataArray = self._data.to_numpy()
        self._dataArray = self._data.get_values()


        return self._labelArray, self._dataArray

    def divideLabelAndData(self, dataColRanges, labelCol):
        """
        Input:
            dataColRange: start and end name of data rows, start & end name are tuple pairs, names are string
            labelCol: string. You can only pass dataframe with one col of data!
        Return:
            dataframe of data(as one) & label

        Alert:
            If you pass a pair with wrong order: 


        This method only for this very specific example, it's not a good implementation!
        """

        # if dataColRanges is wrong

        start = dataColRanges[0]
        end = dataColRanges[1]
        self._data = self._frameData.loc[:, start: end]
        self._label = self._frameData.loc[:, labelCol]

        # for debugging
        return self._label, self._data

    def extractLabelInfo(self):
        """
        external
        Find out how many labels types, and return the label value to a list
        """
        label_v = []
        num = 0
        for i, value in self._label.iteritems():
            curr_label = value
            if curr_label in label_v:
                continue
            else:
                label_v.append(value)
                num += 1

        self._nClass = num

        return self._nClass, label_v

    def generateMapping(self):
        """
        Get:
            label-key mapping dictionary
            number of label types
        Notice:
            The value of map start from 0!
        """
        n, v = self.extractLabelInfo()
        self._labelMap = dict({v[i]: i for i in range(n)})  
        # for debugging
        return self._labelMap

    def one_hot_key_encode(self, label_value):
        """
        external
        Input: 
            input label index and total label number
        Return: 
            a one-hot-key encoded sparse list
        """
        label_idx = self._labelMap[label_value]
        origin = np.zeros(self._nClass)
        origin[label_idx] = 1
        return list(origin)

    def one_hot_key_decode(self, label_ohk):
        """
        external
        Input:
            label one-hot-key code array
        Return:
            label string
        Notice:
            This function is meant to use for presentation the result
        """ 
        label_str = [k for k, v in self._labelMap.items() if v == np.argmax(label_ohk)]
        return str(label_str[0])


    def plotLDA(self, figs = (10, 10)):
        fig = plt.figure(figsize = figs)

        lda = LDA(n_components=2) #2-dimensional LDA
        label = self._label
        data = self._data
        lda_transformed = pd.DataFrame(lda.fit_transform(data, label))

        n, v = self.extractLabelInfo()
        color=iter(plt.cm.rainbow(np.linspace(0,1,n)))
        for i in range(n):
            label_i = v[i]
            plt.scatter(lda_transformed[label == label_i][0],lda_transformed[label == label_i][1], label='Class '+ label_i, c=next(color))
        plt.legend(loc = 1)
        plt.title("LDA")
        plt.show()
        return fig

    def plotPCA(self, figs=(10,10)):
        fig = plt.figure(figsize = figs)

        pca = sklearnPCA(n_components=2)
        label = self._label
        data = self._data
        pca_transformed = pd.DataFrame(pca.fit_transform(data))

        n, v = self.extractLabelInfo()
        color=iter(plt.cm.rainbow(np.linspace(0,1,n)))
        for i in range(n):
            label_i = v[i]
            plt.scatter(pca_transformed[label == label_i][0],pca_transformed[label == label_i][1], label='Class '+ label_i, c=next(color))
        plt.legend(loc = 1)
        plt.title("PCA")
        plt.show()
        return fig


if __name__ == "__main__":
    dataColRanges = ('1-ZCRm', '34-ChromaDeviationm')
    labelCol = 'class'
    a = Classifier_dataset("../../data/data_set/beatsdataset.csv")
    a.prepareData(dataColRanges, labelCol, True)
    print(a._dataArray)
    print(a._labelArray)
    fig1 = a.plotPCA()
    fig2 = a.plotLDA()