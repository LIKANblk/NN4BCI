from Data import *
from keras.layers import Convolution1D, Dense, Dropout, Input, merge, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras import backend as K
from scipy.signal import resample
from uuid import uuid4
from convolutions import *
from conv_trees import *
from AlexPreprocess import *


def get_base_model(input_len, fsize,channel_number):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_seq = Input(shape=(input_len, channel_number))
    nb_filters = 50
    convolved = Convolution1D(nb_filters, fsize, border_mode="same", activation="relu")(input_seq)
    pooled = GlobalMaxPooling1D()(convolved)
    compressed = Dense(50, activation="relu")(pooled)
    compressed = Dropout(0.7)(compressed)
    compressed = Dense(50, activation="relu")(compressed)
    compressed = Dropout(0.7)(compressed)
    model = Model(input=input_seq, output=compressed)
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

    model = Model(input=[input_full_seq], output=out)

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

def data_placement(dat,convs,conv_seqs):
    # @data_trial of shape (trials x time x convs)
    # @convs_seqs - np.array of (number of seqs x seq length)
    seq_lenght = conv_seqs.shape[1]
    time_len = (data.shape[1] - (seq_lenght-1))*seq_lenght
    res = np.zeros(data.shape[0],time_len,len(conv_seqs))
    res_time_counter = 0
    for t in range(data_trial.shape[0]-seq_length)
        for dt in range(seq_length):
			res[:,res_time_counter,:] = data[:,t+dt,conv_seqs[:,dt]]
	return res

if __name__=='__main__':

    data_source = NeuromagData('mag')
    dim_order = ['trial','time','channel']
    X,y=data_source.get_all_experiments_data(dim_order)
    dev = Neuromag('mag')
    convs = Convolutions(dev)
    conv_seqs = cc.make_geodesic_conv_combinations(dev.topography_3D, np.array(convs), 3, 0.4, 0.4, 0.3, cc.curr_faces,
                                             'directions_Real.csv')
    conv_seqs = filter(lambda x: x[1] < 0.15, conv_seqs)
    data = AlexPreprocess().extract_expert_features(X,convs)
    
	

    # augmenter = DataAugmentation(device=dev)
    # Xm = augmenter.mirror_sensors(X)
    # X = np.concatenate((X,Xm),axis=0)
    # y = np.hstack((y,y))
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # model = get_full_model(epoch_len = X.shape[1],channel_number = X.shape[2])
    # nb_epoch = 10000
    # early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    # tensor_board = TensorBoard(log_dir = './logs/'+str(uuid4()), histogram_freq = 3)
    # with K.tf.device('/gpu:2'):
    #     model.fit(x=get_resampled_data(X,axis=1),y=to_onehot(y),batch_size=30, nb_epoch = nb_epoch,
    #                         callbacks=[tensor_board], verbose=1, validation_split=0.2,shuffle=True)
