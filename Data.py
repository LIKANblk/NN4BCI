from scipy.io import loadmat
import os
from devices import *
import numpy as np
from random import sample

class Data:
    def __init__(self,path_to_data):
        self.path_to_data = path_to_data

    def load_from_spec_folder(self,path_to_folder):
        #Process all .mat files in specified folder
        return np.concatenate(
            [self.process_mat_file(os.path.join(path_to_folder, f)) for f in os.listdir(path_to_folder) if f.endswith(".mat")], axis=0)

    def process_mat_file(self,path_to_mat):
        #Process specified .mat file
        data = loadmat(path_to_mat)
        return (data['F'])[np.newaxis, ...].astype('float32',
                        casting='same_kind')  # additional dimension for easier concatenation to 3d array in the future


    def data_shuffle(self, x, y):
        d_len = y.shape[0]
        from random import shuffle
        sh_data = range(d_len)
        shuffle(sh_data)
        new_y = np.zeros(y.shape,dtype=np.int)
        for i in range(d_len):
            new_y[i] = y[sh_data[i]]
        res_x = np.zeros(x.shape)
        for i in range(d_len):
            res_x[i] = x[sh_data[i]]
        return  res_x,new_y


class GSN128Data(Data):
    def __init__(self):
        Data.__init__(self,path_to_data = 'DATA/GSN128/')

    def get_data_labels(self):
        is_file = lambda filename: os.path.isfile(os.path.join(self.path_to_data,filename))
        labels = filter(is_file, os.listdir(self.path_to_data))
        labels = [label[:-4] for label in labels] #removing file extension
        return labels

    def get_all_data(self):
        # Process all .mat files in specified folder
        labels = self.get_data_labels()
        res = {}
        for label in labels:
            iteration_path = os.path.join((self.path_to_data,label))
            res[label] = self.process_mat_file(iteration_path)
        return res

    def get_data_by_label(self,label):
        iteration_path = os.path.join(self.path_to_data, label)
        return self.process_mat_file(iteration_path)



class NeuromagData(Data):
    def __init__(self,sensor_type,epoch_start=0,epoch_end =1321,
                 saccade_start=724,saccade_end=920):
        #@sensor_type - 'mag' or 'grad'
        self.sensor_type = 'MEG ' + sensor_type.upper()
        chan_info = loadmat(os.path.join('neuromag_info','ChannelType.mat'))
        self.sensor_mask = np.array(map(lambda x: x[0][0] == self.sensor_type, chan_info['Type']))
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.saccade_start = saccade_start
        self.saccade_end = saccade_end
        if self.sensor_type == 'MEG MAG':
            self.num_channels = 102
        if self.sensor_type == 'MEG GRAD':
            self.num_channels = 204
        self.time_length = (epoch_end - epoch_start) - (saccade_end - saccade_start)
        Data.__init__(self,path_to_data = os.path.join('DATA','Neuromag'))
        self.exp_names = (self.get_experiments_names())


    def get_data_labels(self):
        experiment_name = self.exp_names[0]
        is_dir = lambda filename: os.path.isdir(os.path.join(self.path_to_data,experiment_name,filename))
        return filter(is_dir, os.listdir(os.path.join(self.path_to_data,experiment_name)))

    def get_all_experiments_data(self,target_dim_order=['trial', 'channel', 'time'], label_each_exp=False,normalise=False):
        #label_each_exp - if True target class in each experiment will have unique label (number of experiment)
        def_configuration = ['trial', 'channel', 'time']
        transpose_mask = [def_configuration.index(dim) for dim in target_dim_order]
        X = np.empty((0, self.num_channels, self.time_length), dtype=np.float64).transpose(transpose_mask)
        y = np.empty((0), dtype=np.int)
        for exp_num,exp_name in enumerate(self.exp_names):
            X_tmp,y_tmp = self.get_data_from_exp(exp_name,target_dim_order,normalise)
            X = np.append(X,X_tmp,axis=0)
            if label_each_exp:
                y_tmp *= (exp_num+1)
            y = np.append(y,y_tmp,axis=0)

        return self.data_shuffle(X,y)

    def get_data_from_exp(self,experiment,target_dim_order=['trial', 'channel', 'time'],normalise=False):
        # @dim_order - list of strings in format ['trial','channel','time']
        labels = self.get_data_labels()

        def_configuration = ['trial', 'channel', 'time']
        transpose_mask = [def_configuration.index(dim) for dim in target_dim_order]

        X = np.empty((0, self.num_channels, self.time_length), dtype=np.float64)
        y = np.empty((0), dtype=np.int)
        for index,label in enumerate(labels):
            tmp = self.get_data_by_label(experiment,label)
            X =np.append(X,tmp,axis=0)
            y = np.append(y,np.full(tmp.shape[0],index),axis=0)

        if normalise:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X = X.transpose(transpose_mask)
        return X,y

    def get_time_mask(self,interval_length):
        time_mask = np.full(interval_length, False)
        time_mask[self.epoch_start:self.epoch_end] = True
        time_mask[self.saccade_start:self.saccade_end] = False
        return time_mask

    def get_data_by_label(self,experiment,label):
        #@dim_order - list of strings in format ['trial','channel','time']
        #@return numpy array (trial,channel,time)
        data_by_label_path = os.path.join(self.path_to_data,experiment,label)
        data = self.load_from_spec_folder(data_by_label_path)[:, self.sensor_mask, :]

        time_mask = self.get_time_mask(data.shape[2])
        return data[:,:,time_mask]

    def get_experiments_names(self):
        is_dir = lambda filename: os.path.isdir(os.path.join(self.path_to_data, filename))
        return sorted(filter(is_dir, os.listdir(self.path_to_data)))

    def get_all_names(self,val,val_split=0.2):
        labels = self.get_data_labels()
        # res = {exp_index:[] for exp_index in range(len(exp_names)+1)}
        res = []
        list_dir = lambda path,label: sorted([(os.path.join(path,f),label) for f in os.listdir(path) if f.endswith(".mat")])
        for exp_index,exp_name in enumerate(self.exp_names):
            path_to_target = os.path.join(self.path_to_data,exp_name,labels[1])
            path_to_nontarget = os.path.join(self.path_to_data, exp_name, labels[0])
            target = list_dir(path_to_target,exp_index+1)
            nontarget = list_dir(path_to_nontarget,0)
            if val:
                target = target[-int(np.ceil(len(target) * (1 - val_split))):]
                nontarget = nontarget[-int(np.ceil(len(nontarget) * (1-val_split))):]
            else:
                target = target[:int(np.floor(len(target)*(1-val_split)))]
                nontarget = nontarget[:int(np.floor(len(nontarget) * (1-val_split)))]
            res +=target
            res += nontarget
        return res

    def read_mat_from_files(self,files):
        return np.concatenate([self.process_mat_file(f) for f in files], axis=0)

    def get_random_batch(self,batchsize,all_data_files,target_dim_order,normalise=False,label_each_exp=True):
        def to_onehot(y):
            onehot = np.zeros((len(y), len(self.exp_names)+1))
            if not label_each_exp:
                y=[int(elem>0) for elem in y]
            onehot[range(len(y)), y] = 1
            return onehot
        indexes = sample(range(len(all_data_files)),batchsize)
        files = [all_data_files[i][0] for i in indexes]
        labels = np.array([all_data_files[i][1] for i in indexes])
        batch = self.read_mat_from_files(files)
        time_mask = self.get_time_mask(batch.shape[-1])
        batch = batch[:,:,time_mask]
        batch = batch[:, self.sensor_mask, :]
        if normalise:
            batch = (batch - np.mean(batch, axis=0)) / np.std(batch, axis=0)

        def_configuration = ['trial', 'channel', 'time']
        transpose_mask = [def_configuration.index(dim) for dim in target_dim_order]

        return batch.transpose(transpose_mask),to_onehot(labels)

    def get_batch_generator(self,batchsize,all_data_files,target_dim_order,normalise=False,label_each_exp=True):

        while True:
            yield self.get_random_batch(batchsize, all_data_files, target_dim_order,normalise,label_each_exp)

class DataAugmentation:
    def __init__(self,device):
        self.device = device

    def mirror_sensors(self,data):
        def find_symmetric():
            channels = range(len(self.device.ch_names))
            topography_3D = self.device.topography_3D
            left, right = [], []
            for ch in channels:
                if topography_3D[ch, 0] < 0:
                    left.append(ch)
                if topography_3D[ch, 0] > 0:
                    right.append(ch)
            res = []
            for l in left:
                tmp = [[l, r] for r in right]
                key_func = lambda pair: np.linalg.norm(topography_3D[pair[0]][1:] - topography_3D[pair[1]][1:])
                sym_pair = min(tmp, key=key_func)
                #         print cn.ch_names[sym_pair[0]],cn.ch_names[sym_pair[1]]
                right.remove(sym_pair[1])
                res.append(sym_pair)
            return res
        ch_dim = data.shape.index(self.device.num_channels)
        res = np.array(data)
        pairs = find_symmetric()
        for pair in pairs:
            source_indices = [slice(None)]*ch_dim + [pair] + [slice(None)]*(data.ndim-ch_dim-1)
            res_indices = [slice(None)]*ch_dim + [pair[::-1]] + [slice(None)]*(data.ndim-ch_dim-1)
            res[res_indices] = res[source_indices]
        return res

if __name__ == '__main__':
    dev = Neuromag('mag')
    data_source = NeuromagData('mag')
    res = data_source.get_all_names()

    while True:
        batch,y = next(data_source.get_batch_generator(batchsize=50, all_data_files = res,target_dim_order=['trial', 'time', 'channel']))
    print 'ok'
    # augmenter = DataAugmentation(device=dev)
    # mirrored_data = augmenter.mirror_sensors(data)
