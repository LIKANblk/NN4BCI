from scipy.io import loadmat
from os import listdir
import os.path
import numpy as np


class Data:
    def __init__(self,path_to_data):
        self.path_to_data = path_to_data


    def get_all_data(self):
        #Loads ALL data from ALL folders for ALL classes
        print 'Not implementer'
        pass

    def load_from_spec_folder(self,path_to_folder):
        #Process all .mat files in specified folder
        return np.concatenate(
            [self.process_mat_file(os.path.join(path_to_folder, f)) for f in listdir(path_to_folder) if f.endswith(".mat")], axis=0)

    def process_mat_file(self,path_to_mat):
        #Process specified .mat file
        data = loadmat(path_to_mat)
        return (data['F'])[np.newaxis, ...].astype('float32',
                        casting='same_kind')  # additional dimension for easier concatenation to 3d array in the future

class GSN128Data(Data):
    def __init__(self):
        Data.__init__(self,path_to_data = 'DATA/synthetic/')

    def load_all_data(self):
        #Process all .mat files in specified folder
        path_to_folder = self.path_to_data
        return self.load_from_spec_folder(path_to_folder)



# class NeuromagData(Data):
#     def __init__(self,sensor_type):
#         self.path_to_data = 'DATA/experimental/'
#         self.sensor_type = sensor_type
#         self.channel_types = loadmat('neuromag_info/ChannelType.mat')
#
#         Data.__init__()
#
#
#     def load_data(path, sensor_type):
#         # This function load data from particular folder
#         #
#         # @sensor_type -  'MEG GRAD' or 'MEG MAG'
#         mask_rawdata = loadmat('ChannelType.mat')
#         gradiom_mask = np.array(map(lambda x: x[0][0] == sensor_type, mask_rawdata['Type']))
#         return np.concatenate(
#             [extract_grad_mat(join(path, f), gradiom_mask) for f in listdir(path) if f.endswith(".mat")], axis=0)
#
#     def extract_grad_mat(path, gradiom_mask):
#         data = loadmat(path)
#         return (data['F'][gradiom_mask])[np.newaxis, ...].astype('float32',
#                                                                  casting='same_kind')  # additional dimension for easier concatenation to 3d array in the future
#     def get_data(path, sensor_type):
#         # This function defines path to target and non-target (using particular labels) data and loads both of them
#         # @sensor_type -  'MEG GRAD' or 'MEG MAG'
#         path_to_target = join(path, 'SI')
#         path_to_nontarget = join(path, 'error')
#         target_data = load_data(path_to_target, sensor_type)
#         nontarget_data = load_data(path_to_nontarget, sensor_type)
#         return target_data, nontarget_data
