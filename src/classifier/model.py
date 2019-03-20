# from keras.models import Sequential
# from keras.layers import Dense
import keras
import numpy as np
import prepareData

class MusicClassifier:
    """
    This class is used for applying the exist model to the project
    It has to be loaded from a existing model (file or keras model object)
    You can retrain 

    _model : keras model loaded from .h5 file



    """


    def __init__(self, model):
        """
        init from keras model obj
        """
        self._model = model
        return self._model

    def loadModel(self, filename, type = 'w'):
        """
        type:
            'm' : whole model
            'l' : model's layer params
            'w' : model's weight

        right now only support .h5 file
        """
        if type == 'w':
            self._model = keras.models.load_model(filename)

        return self._model


    

    def trianModel(self, filename):
        """
        Load .h5 file, extract the model

        """
        frame_data = MusicClassifier(filename)
        # self._dataset = 
        pass


    
    
    def saveModel(self, filename, type = 'm'):
        """
        type:
            'm' : whole model
            'l' : model's layer params
            'w' : model's weight

        .h5 is auto added
        layer param is json file
        """
        # if type == 'l':
        #     self._model.to_json()
        # elif type == 'w':
        #     self._model.save_weights(filename + 'h5')
        # elif type == 'm':
        #     self._model.save(filename + 'h5')
        # else:
        #     print("wrong type")
        #     return
        pass



    def predict(self, data):
        """
        data is the output of last layer of model, ndarray
        """
        softmax_o = self._model.predict(data)
        max_ele = max(softmax_o)
        ohk_out = np.where(softmax_o >= max_ele, 1, 0)
        result = prepareData.Classifer_dataset.one_hot_key_decode(ohk_out)
        print("Huh, this song sounds like " + result)
        




if __name__ == "__main__":
    # filename = "./example.h5"
    # a = MusicClassifier.loadModel(filename)

    # example = np.random.random_sample(23)
    # a.predict(example)
    # b = MusicClassifier(model)
    a = Classifer_dataset("../../data/data_set/beatsdataset.csv")
    pass
    