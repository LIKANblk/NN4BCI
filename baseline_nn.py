from Data import *
from keras.layers import Convolution1D, Dense, Dropout, Input, merge, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop

def get_base_model(input_len, fsize,channel_number):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_seq = Input(shape=(input_len, channel_number))
    nb_filters = 50
    convolved = Convolution1D(nb_filters, fsize, border_mode="same", activation="tanh")(input_seq)
    pooled = GlobalMaxPooling1D()(convolved)
    compressed = Dense(50, activation="linear")(pooled)
    compressed = Dropout(0.3)(compressed)
    compressed = Dense(50, activation="relu")(compressed)
    compressed = Dropout(0.3)(compressed)
    model = Model(input=input_seq, output=compressed)
    return model

def apply_model(channel_number):
    input256_seq = Input(shape=(256, channel_number))
    input500_seq = Input(shape=(500, channel_number))
    input1125_seq = Input(shape=(1125, channel_number))

    base_network256 = get_base_model(256, 4)
    base_network500 = get_base_model(500, 7)
    base_network1125 = get_base_model(1125, 10)

    embedding_256 = base_network256(input256_seq)
    embedding_500 = base_network500(input500_seq)
    embedding_1125 = base_network1125(input1125_seq)

    merged = merge([embedding_256, embedding_500, embedding_1125], mode="concat")
    out = Dense(3, activation='softmax')(merged)

    model = Model(input=[input256_seq, input500_seq, input1125_seq], output=out)

    opt = RMSprop(lr=0.005, clipvalue=10**6)
    model.compile(loss="categorical_crossentropy", optimizer=opt)



if __name__ == 'main':
    experiment='em06'
    data = NeuromagData('mag')
    for index,label in enumerate(data.get_data_labels()):
        X = data.get_data_by_label(experiment=experiment,label=label)
        y = np.full(X.shape[0],index)
