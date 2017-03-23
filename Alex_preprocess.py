import numpy as np
from scipy.linalg import norm
from scipy.signal import hilbert
from convolutions import convolution_neuromag
from itertools import combinations
from load_data import Data

class Alex_preprocess:
    def __init__(self,sensor_type, conv_length):
        self.conv_length
        pass

    def extract_expert_features(self,conv_length):
        data_object = Data()
        self.data = data_object.get_data()
        conv_object = convolution_neuromag('mag')
        self.convolution_indeces = conv_object.get_1Dconvolutions(conv_length)
        result = np.zeros()

    def data_generator(self,data):
        #@data - (channel x time) i.e. ONE TRIAL
        for conv in self.convolution_indeces:
            yield data[conv,:]

    def _calc_local_phase(self):
        analytic_signal = hilbert(self.data)
        return np.unwrap(np.angle(analytic_signal))

    def _get_phase_delta(self,conv_values):
        for i,j in combinations(conv_values,2):
            return i-j

    def _activation_function(self,beta,x):
        return np.exp(-beta*norm(x)**2)
