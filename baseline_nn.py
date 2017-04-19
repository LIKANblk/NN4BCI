#TODO 1) Devide data to batches
#TODO 2)
from keras.models import Model
from keras.layers.convolutional import Convolution1D
from keras.layers import merge, Input
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import GlobalAveragePooling1D

def get_base_model():
    '''Base network to be shared (eq. to feature extraction).
    '''
    with K.tf.device('/gpu:2'):
        input_seq = Input(shape=(1125, 24))

        filter_sizes = [5, 7, 14]
        nb_filters = 100
        #         filter_size = 7
        different_scales = []
        for fsize in filter_sizes:
            convolved = Convolution1D(nb_filters, fsize, border_mode="same", activation="tanh")(input_seq)
            processed = GlobalMaxPooling1D()(convolved)
            different_scales.append(processed)

        different_scales = merge(different_scales, mode='concat')
        compressed = Dense(150, activation="tanh")(different_scales)
        compressed = Dropout(0.2)(compressed)
        compressed = Dense(150, activation="tanh")(compressed)
        model = Model(input=input_seq, output=compressed)
        return model