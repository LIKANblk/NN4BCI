import numpy as np

class Data:
    def __init__(self,):
        pass
    def get_data(self):  #TODO
        return np.random.rand(100,102,400) #temporary stub (trials x channels x time)
    # def get_convovled_data(self,convolutions):
    #     for conv in self.convolution_indeces:
    #         yield self.data[:,conv,:]