from Data import *
from keras.layers import Conv1D, Dense, Dropout, Input, GlobalMaxPooling1D, GaussianNoise
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras import backend as K
from scipy.signal import resample
from uuid import uuid4
from convolutions import *
import conv_trees as cc
from AlexPreprocess import *
from itertools import imap


def get_base_model(input_len, fsize,feature_number,seq_length):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_seq = Input(shape=(input_len, feature_number))
    state0 = GaussianNoise(0.01)(input_seq)
    state1 = Dropout(0.3)(state0)
    nb_filters = 40
    convolved = Conv1D(nb_filters, fsize, padding="same", activation="relu",strides=seq_length)(state1)
    pooled = GlobalMaxPooling1D()(convolved)
    compressed = Dense(30, activation="relu")(pooled)
    compressed = Dropout(0.4)(compressed)
    compressed = Dense(20, activation="relu")(compressed)
    compressed = Dropout(0.4)(compressed)
    model = Model(inputs=input_seq, outputs=compressed)
    return model

def get_full_model(epoch_len,feature_number,seq_length,classes_num):
    input_seq = Input(shape=(epoch_len, feature_number))


    base_network_full = get_base_model(epoch_len, fsize=seq_length,feature_number=feature_number,seq_length = seq_length)

    embedding_full = base_network_full(input_seq)

    # merged = merge([embedding_quarter, embedding_half, embedding_full], mode="concat")
    out = Dense(classes_num, activation='softmax')(embedding_full)

    model = Model(inputs=[input_seq], outputs=out)

    opt = RMSprop(lr=0.00005, clipvalue=10**6)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=opt)
    return model


def to_onehot(y,num_classes):
    onehot = np.zeros((len(y),num_classes))
    onehot[range(len(y)),list(y)] = 1
    return onehot


def data_placement(data,conv_seqs):
    # @data_trial of shape (trials x time x convs)
    # @convs_seqs - np.array of (number of seqs x seq length)
    seq_length = conv_seqs.shape[1]
    time_len = (data.shape[1] - (seq_length-1))*seq_length
    res = np.zeros((data.shape[0],time_len,conv_seqs.shape[0]))
    res_time_counter = 0
    for t in range(data.shape[1]-(seq_length-1)):
        for dt in range(seq_length):
            res[:,res_time_counter,:] = (data[:,t+dt,:])[:,conv_seqs[:,dt]]
            res_time_counter+=1
    return res

def full_data_approach():
    # RAM consuming
    data_source = NeuromagData('mag')
    dim_order = ['trial', 'time', 'channel']
    X, y = data_source.get_all_experiments_data(target_dim_order=dim_order, label_each_exp=True)
    num_of_classes = len(data_source.get_exeriments_names()) + 1
    dev = Neuromag('mag')
    convs = Convolutions(dev).get_1Dconvolutions(conv_length=3)
    conv_seqs = cc.make_geodesic_conv_combinations(dev.topography_3D, np.array(convs), 3, 0.4, 0.4, 0.3, cc.curr_faces,
                                                   'directions_Real.csv')
    conv_seqs = np.array([elem[0] for elem in conv_seqs if (elem[1] < 0.15)])
    X = AlexPreprocess().extract_expert_features(X, convs, ch_dim=2, beta=0.1)
    X = data_placement(X, conv_seqs)
    model = get_full_model(epoch_len=X.shape[1], feature_number=X.shape[2], seq_length=conv_seqs.shape[1],
                           classes_num=num_of_classes)
    nb_epoch = 10000
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    tensor_board = TensorBoard(log_dir='./logs/' + str(uuid4()), histogram_freq=3)
    with K.tf.device('/gpu:2'):
        model.fit(x=X, y=to_onehot(y, num_of_classes), batch_size=20, nb_epoch=nb_epoch,
                  callbacks=[tensor_board], verbose=1, validation_split=0.2, shuffle=True)
    print 'ok'

def generator_approach():
    # Generator approach for large amount of data

    dim_order = ['trial', 'time', 'channel']

    data_source = NeuromagData('mag')
    num_of_classes = len(data_source.exp_names) + 1
    dev = Neuromag('mag')
    convs = Convolutions(dev).get_1Dconvolutions(conv_length=3)
    conv_seqs = cc.make_geodesic_conv_combinations(dev.topography_3D, np.array(convs), 3, 0.4, 0.4, 0.3, cc.curr_faces,
                                                   'directions_Real.csv')
    conv_seqs = np.array([elem[0] for elem in conv_seqs if (elem[1] < 0.15)])

    f = lambda x,:data_placement(AlexPreprocess().extract_expert_features(x, convs, ch_dim=2, beta=0.1), conv_seqs)
    train_gen = imap(lambda x: (f(x[0]),x[1]),
                   data_source.get_batch_generator(50, data_source.get_all_names(val=False),
                                                   target_dim_order=dim_order,normalise=False,label_each_exp=True))

    val_gen = imap(lambda x: (f(x[0]), x[1]),
                   data_source.get_batch_generator(50, data_source.get_all_names(val=True),
                                                   target_dim_order=dim_order,normalise=False,label_each_exp=True))
    num_seqs = conv_seqs.shape[0]
    seq_length = conv_seqs.shape[1]
    data_source = NeuromagData('mag')
    time_length =(data_source.time_length - (seq_length-1))*seq_length
    model = get_full_model(epoch_len=time_length,
                           feature_number=num_seqs, seq_length=seq_length,
                           classes_num=num_of_classes)
    nb_epoch = 10000
    # early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    tensor_board = TensorBoard(log_dir='./logs/' + str(uuid4()), histogram_freq=3)
    with K.tf.device('/gpu:2'):
        model.fit_generator(generator=train_gen, steps_per_epoch=20, epochs=nb_epoch,
                            callbacks=[tensor_board], verbose=1,validation_steps=4,validation_data=val_gen)
    print 'ok'

if __name__=='__main__':
    generator_approach()

