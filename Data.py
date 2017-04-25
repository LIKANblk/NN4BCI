from scipy.io import loadmat
import os
from devices import *
import numpy as np


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



    def get_data_labels(self):
        experiment_name = self.get_exeriments_names()[0]
        is_dir = lambda filename: os.path.isdir(os.path.join(self.path_to_data,experiment_name,filename))
        return filter(is_dir, os.listdir(os.path.join(self.path_to_data,experiment_name)))


    def get_all_data(self,experiment,target_dim_order=['trial', 'channel', 'time']):
        # @dim_order - list of strings in format ['trial','channel','time']
        labels = self.get_data_labels()

        def_configuration = ['trial', 'channel', 'time']
        transpose_mask = [def_configuration.index(dim) for dim in target_dim_order]

        X = np.empty((0, self.num_channels, self.time_length), dtype=np.float64).transpose(transpose_mask)
        y = np.empty((0), dtype=np.int)
        for index,label in enumerate(labels):
            tmp = self.get_data_by_label(experiment,label).transpose(transpose_mask)
            X =np.append(X,tmp,axis=0)
            y = np.append(y,np.full(tmp.shape[0],index),axis=0)
        return X,y

    def get_data_by_label(self,experiment,label):
        #@dim_order - list of strings in format ['trial','channel','time']
        #@return numpy array (trial,channel,time)
        def get_time_mask(interval_length):
            time_mask = np.full(interval_length, False)
            time_mask[self.epoch_start:self.epoch_end] = True
            time_mask[self.saccade_start:self.saccade_end] = False
            return time_mask

        data_by_label_path = os.path.join(self.path_to_data,experiment,label)
        data = self.load_from_spec_folder(data_by_label_path)[:, self.sensor_mask, :]

        time_mask = get_time_mask(data.shape[2])
        return data[:,:,time_mask]

    def get_exeriments_names(self):
        is_dir = lambda filename: os.path.isdir(os.path.join(self.path_to_data, filename))
        return filter(is_dir, os.listdir(self.path_to_data))

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
    data,label = data_source.get_all_data('em01',target_dim_order = ['trial','time','channel'])
    augmenter = DataAugmentation(device=dev)
    mirrored_data = augmenter.mirror_sensors(data)
