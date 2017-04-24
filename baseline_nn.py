from Data import *
from keras.layers import Convolution1D, Dense, Dropout, Input, merge, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K
from scipy.signal import resample

def get_base_model(input_len, fsize,channel_number):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_seq = Input(shape=(input_len, channel_number))
    nb_filters = 200
    convolved = Convolution1D(nb_filters, fsize, border_mode="same", activation="tanh")(input_seq)
    pooled = GlobalMaxPooling1D()(convolved)
    compressed = Dense(250, activation="linear")(pooled)
    compressed = Dropout(0.3)(compressed)
    compressed = Dense(250, activation="relu")(compressed)
    compressed = Dropout(0.3)(compressed)
    model = Model(input=input_seq, output=compressed)
    return model

def get_full_model(channel_number):
    input256_seq = Input(shape=(256, channel_number))
    input500_seq = Input(shape=(500, channel_number))
    input1125_seq = Input(shape=(1125, channel_number))

    base_network256 = get_base_model(256, 4,channel_number)
    base_network500 = get_base_model(500, 7,channel_number)
    base_network1125 = get_base_model(1125, 10,channel_number)

    embedding_256 = base_network256(input256_seq)
    embedding_500 = base_network500(input500_seq)
    embedding_1125 = base_network1125(input1125_seq)

    merged = merge([embedding_256, embedding_500, embedding_1125], mode="concat")
    out = Dense(2, activation='softmax')(merged)

    model = Model(input=[input256_seq, input500_seq, input1125_seq], output=out)

    opt = RMSprop(lr=0.000005, clipvalue=10**6)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=opt)
    return model


def to_onehot(y):
    onehot = np.zeros((len(y),2))
    onehot[range(len(y)),y] = 1
    return onehot

def get_resampled_data(data,axis):
    return [resample(data,256,axis=axis),resample(data,500,axis=axis),data]
if __name__=='__main__':
    experiment='em02'
    data = NeuromagData('mag')
    dim_order = ['trial','time','channel']
    X,y=data.get_all_data(experiment,dim_order)


    model = get_full_model(X.shape[2])
    nb_epoch = 1000
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10000, verbose=0, mode='auto')

    with K.tf.device('/gpu:2'):
        model.fit(x=get_resampled_data(X,1),y=to_onehot(y),batch_size=300, nb_epoch = nb_epoch,
                            callbacks=[earlyStopping], verbose=1, validation_split=0.2)
