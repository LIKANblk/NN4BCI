from Data import *
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, merge,GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras import backend as K
from scipy.signal import resample
from uuid import uuid4
from cross_features import CrossFeatures
import sys

# def get_base_model(input_len, fsize,feature_num, feat_ch=2):
#     '''Base network to be shared (eq. to feature extraction).
#     '''
#     input_seq = Input(shape=(input_len, feature_num, feat_ch))
#     nb_filters = 50
#     convolved = Conv2D(nb_filters, nb_row=feature_num,nb_col=fsize, border_mode="same", activation="relu",dim_ordering="tf")(input_seq)
#     # convolved = Dropout(0.7)(convolved)
#     pooled = GlobalMaxPooling2D(dim_ordering='tf')(convolved)
#     compressed = Dense(50, activation="relu")(pooled)
#     compressed = Dropout(0.7)(compressed)
#     compressed = Dense(50, activation="relu")(compressed)
#     compressed = Dropout(0.7)(compressed)
#     model = Model(input=input_seq, output=compressed)
#     return model
#
# def get_full_model(epoch_len,feature_num,feat_ch=2):
#     # input_quarter_seq = Input(shape=(int(epoch_len/4), channel_number))
#     # input_half_seq = Input(shape=(int(epoch_len/2), channel_number))
#     input_full_seq = Input(shape=(epoch_len, feature_num,feat_ch))
#
#     # base_network_quarter = get_base_model(int(epoch_len/4), 10,channel_number)
#     # base_network_half = get_base_model(int(epoch_len/2), 10,channel_number)
#     base_network_full = get_base_model(input_len=epoch_len, fsize = 10,feature_num=feature_num)
#
#     # embedding_quarter = base_network_quarter(input_quarter_seq)
#     # embedding_half = base_network_half(input_half_seq)
#     embedding_full = base_network_full(input_full_seq)
#
#     # merged = merge([embedding_quarter, embedding_half, embedding_full], mode="concat")
#     out = Dense(2, activation='softmax')(embedding_full)
#
#     model = Model(input=[input_full_seq], output=out)
#
#     opt = RMSprop(lr=0.00005, clipvalue=10**6)
#     model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=opt)
#     return model

def get_base_model(input_len,fsize ,feature_num,dr):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_seq = Input(shape=(input_len, feature_num))
    nb_filters = 40
    convolved = Conv1D(nb_filters, kernel_size=fsize, padding="same", activation="relu",)(input_seq)
    pooled = GlobalMaxPooling1D()(convolved)
    compressed = Dense(30, activation="relu")(pooled)
    compressed = Dropout(dr)(compressed)
    compressed = Dense(20, activation="relu")(compressed)
    compressed = Dropout(dr)(compressed)
    model = Model(inputs=input_seq, outputs=compressed)
    return model

def get_full_model(epoch_len,feature_num,dr,lr):
    # input_quarter_seq = Input(shape=(int(epoch_len/4), feature_num))
    # input_half_seq = Input(shape=(int(epoch_len/2), feature_num))
    input_full_seq = Input(shape=(epoch_len, feature_num))

    # base_network_quarter = get_base_model(int(epoch_len/4), 10,feature_num=feature_num,dr=dr)
    # base_network_half = get_base_model(int(epoch_len/2), 10,feature_num=feature_num,dr=dr)
    base_network_full = get_base_model(input_len=epoch_len, fsize = 10,feature_num=feature_num,dr=dr)

    # embedding_quarter = base_network_quarter(input_quarter_seq)
    # embedding_half = base_network_half(input_half_seq)
    embedding_full = base_network_full(input_full_seq)

    # merged = Concatenate()([embedding_quarter, embedding_half, embedding_full])
    out = Dense(2, activation='softmax')(embedding_full)

    model = Model(inputs=[input_full_seq], outputs=out)

    opt = RMSprop(lr=lr, clipvalue=10**6)
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



if __name__=='__main__':
    dev = Neuromag('mag')
    dim_order = ['trial', 'time', 'channel']
    data_source = NeuromagData('mag')
    X, y = data_source.get_all_experiments_data(dim_order,normalise=True)
    augmenter = DataAugmentation(device=dev)
    Xm = augmenter.mirror_sensors(X)
    X = np.concatenate((X,Xm),axis=0)
    y = np.hstack((y,y))
    cf = CrossFeatures(dev)
    features = cf.extract_expert_features(X, conv_length=3).reshape((X.shape[0],X.shape[1],-1))
    lr = float(sys.argv[1])
    dr = float(sys.argv[2])
    model = get_full_model(epoch_len=features.shape[1],feature_num = features.shape[2],lr=lr,dr=dr)
    nb_epoch = 400
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    tensor_board = TensorBoard(log_dir='./logs/' + ('cross_conv_%f_%f' % (lr,dr)), histogram_freq=3)
    with K.tf.device('/gpu:2'):
        sys.stdout = open('cross_conv_%f_%f' % (lr,dr), 'w')
        print 'start'
        model.fit(x=features, y=to_onehot(y), batch_size=30, epochs=nb_epoch,
                  callbacks=[tensor_board], verbose=2, validation_split=0.2, shuffle=True)
