from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
import numpy as np
class Generator:
    def __init__(self):
        '''
        A dummy variable since for some reason python doesn not allow class without
        variabels
        '''
        self.dummy = None
        '''
        Assumes the input is a 2-d array. In order to load in one of the files
        that we are to test our resulting model on, we need to call the numpy method
        np.loadtxt(path_to_file)
        '''
    def rescale(self,input_img):
        '''
        This is going to be a grotesque overexplanation of a single line of code
        but here is how it works:
        
        Firstly, we call the expand dimensions method from numpy. This is done so that
        the array is in the correct format that the keras classifier expects. Secondly,
        in order to convert from 1-channel data to a 3-channel picture, the most elegant
        solution found was the numpy stack method.
        
        According to the numpy documentation, it works as follows:
        join a sequence of arrays along a new axis. The axis argument
        specifies the index of the new axis in the dimensions of the result.
        if 0, it is the first dimension. If -1, it is the last dimension
        
        If you would like to convince yourself that it works, firstly read in
        a training picture as an array, then read in one of the textfiles and call
        this method. Then call the numpy method shape. THey should be exactly the same.
        
        '''
        return np.expand_dims(np.stack((input_img,)*3, axis=-1), axis=0)
        #return np.stack((input_img,)*3, axis=-1)
        
