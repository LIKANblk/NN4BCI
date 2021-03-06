from Data import *
from keras.layers import Conv1D, Dense, Dropout, Input, merge, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras import backend as K
from scipy.signal import resample
from uuid import uuid4


def get_base_model(input_len, fsize,channel_number):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_seq = Input(shape=(input_len, channel_number))
    nb_filters = 50
    convolved = Conv1D(nb_filters, fsize, padding="same", activation="relu")(input_seq)
    convolved = Dropout(0.7)(convolved)
    pooled = GlobalMaxPooling1D()(convolved)
    compressed = Dense(50, activation="relu")(pooled)
    compressed = Dropout(0.7)(compressed)
    compressed = Dense(50, activation="relu")(compressed)
    compressed = Dropout(0.7)(compressed)
    model = Model(inputs=input_seq, outputs=compressed)
    return model

def get_full_model(epoch_len,channel_number):
    # input_quarter_seq = Input(shape=(int(epoch_len/4), channel_number))
    # input_half_seq = Input(shape=(int(epoch_len/2), channel_number))
    input_full_seq = Input(shape=(epoch_len, channel_number))

    # base_network_quarter = get_base_model(int(epoch_len/4), 10,channel_number)
    # base_network_half = get_base_model(int(epoch_len/2), 10,channel_number)
    base_network_full = get_base_model(epoch_len, fsize=10,channel_number=channel_number)

    # embedding_quarter = base_network_quarter(input_quarter_seq)
    # embedding_half = base_network_half(input_half_seq)
    embedding_full = base_network_full(input_full_seq)

    # merged = merge([embedding_quarter, embedding_half, embedding_full], mode="concat")
    out = Dense(2, activation='softmax')(embedding_full)

    model = Model(inputs=[input_full_seq], outputs=out)

    opt = RMSprop(lr=0.00005, clipvalue=10**6)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=opt)
    return model


def to_onehot(y):
    onehot = np.zeros((len(y),2))
    onehot[range(len(y)),y] = 1
    return onehot

def get_resampled_data(data,axis):
    # @axis - axis of time to resempling
    epoch_len = data.shape[axis]
    return [resample(data,epoch_len/4,axis=axis),resample(data,epoch_len/2,axis=axis),data]

def generator_approach():
    data_source = NeuromagData('mag')
    from itertools import imap
    dev = Neuromag('mag')
    def data_augmenter(X,y):
        augmenter = DataAugmentation(device=dev)
        Xm = augmenter.mirror_sensors(X)
        X = np.concatenate((X,Xm),axis=0)
        y = np.vstack((y,y))
        return X,y

    dim_order = ['trial', 'time', 'channel']
    new_gen = imap(lambda x:data_augmenter(*x),data_source.get_batch_generator(50,data_source.get_all_names(),target_dim_order=dim_order))

    model = get_full_model(epoch_len=1125, channel_number=102)
    nb_epoch = 10000
    tensor_board = TensorBoard(log_dir='./logs/' + str(uuid4()), histogram_freq=3)
    with K.tf.device('/gpu:2'):
        model.fit_generator(new_gen, steps_per_epoch=200, epochs=nb_epoch,
                  callbacks=[tensor_board], verbose=2)

if __name__=='__main__':
    # data_source = NeuromagData('mag')
    # dim_order = ['trial','time','channel']
    # X,y=data_source.get_all_experiments_data(dim_order)
    # dev = Neuromag('mag')
    # augmenter = DataAugmentation(device=dev)
    # Xm = augmenter.mirror_sensors(X)
    # X = np.concatenate((X,Xm),axis=0)
    # y = np.hstack((y,y))
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # model = get_full_model(epoch_len = X.shape[1],channel_number = X.shape[2])
    # nb_epoch = 10000
    # early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
    # tensor_board = TensorBoard(log_dir = './logs/'+str(uuid4()), histogram_freq = 3)
    # with K.tf.device('/gpu:2'):
    #     model.fit(x=get_resampled_data(X,axis=1),y=to_onehot(y),batch_size=30, nb_epoch = nb_epoch,
    #                         callbacks=[tensor_board], verbose=1, validation_split=0.2,shuffle=True)
    generator_approach()
