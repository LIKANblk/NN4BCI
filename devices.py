import os
from mne.io import read_raw_fif
from mne.datasets import sample
from mne.channels import read_layout, find_layout,read_ch_connectivity
import numpy as np
from scipy.io import loadmat

class Device(object):
    def __init__(self,ch_names,topography_3D,topography_2D,neighboring,num_channels):
        self.neighboring = neighboring
        self.topography_3D = topography_3D
        self.topography_2D = topography_2D
        self.ch_names = ch_names
        self.num_channels = num_channels

class Neuromag(Device):
    def __init__(self,sensor_type):
        # sensor_type - mag or grad. Grad not yet implemented

        code_dir = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(code_dir, 'neuromag_info')
        neuromag = read_raw_fif(sample.data_path() +
                                '/MEG/sample/sample_audvis_raw.fif')
        topography_2D = find_layout(neuromag.info, ch_type=sensor_type).pos
        if (sensor_type == 'mag'):
            topography_3D = np.array([ch['loc'][:3] for ch in neuromag.info['chs'] if
                                      (ch['ch_name'][-1] == '1') &
                                      (ch['ch_name'][0:3] == 'MEG')]) #Numbers of mag sensors ended by '1'

            neighboring_filename = os.path.join(info_path, 'neuromag306mag_neighb.mat')

        if (sensor_type == 'grad'):
            topography_3D = np.array([ch['loc'][:3] for ch in neuromag.info['chs'] if
                                      ((ch['ch_name'][-1] == '2') | (ch['ch_name'][-1] == '3')) &
                                      (ch['ch_name'][0:3] == 'MEG')])  # Numbers of grad sensors ended by '2' or '3'
            neighboring_filename = os.path.join(info_path, 'neuromag306planar_neighb.mat')

        neighboring, ch_names = read_ch_connectivity(neighboring_filename,
                                                     picks=None)  # ch. names written  in 'MEG1111' format
        neighboring = neighboring.toarray()
        num_channels = len(ch_names)
        self.sensor_type = sensor_type
        Device.__init__(self,ch_names, topography_3D, topography_2D, neighboring, num_channels)

class GSN128(Device):
    def __init__(self):
        code_dir = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(code_dir, 'gsn128_info')
        neighboring_filename = os.path.join(info_path, 'gsn128_neighb.mat')
        topography_2D = read_layout('GSN-128.lay', info_path).pos[:, :2]
        num_channels, ch_names, topography_3D = self._parse_mat(os.path.join(info_path, 'bs_topography.mat'))

        neighboring, self.ch_names = read_ch_connectivity(neighboring_filename, picks=None)
        neighboring = neighboring.toarray()
        Device.__init__(self, ch_names, topography_3D, topography_2D, neighboring, num_channels)

    def _parse_mat(self,mat_file):
        #This function parses topogpraphy .mat from brainstorm. It's really weird magic
        #TODO rewrite search inside imported mat file using read_ch_connectivity mne code
        mat = loadmat(mat_file)
        var_name = mat.keys()[1]

        num_channels = len(mat[var_name][0][0][8][0])
        ch_names = []
        topography_3D = np.zeros((num_channels,3))
        for ch_ind in xrange(num_channels):
            ch_names.append(mat[var_name][0][0][8][0][ch_ind][5][0])
            topography_3D[ch_ind,:]= mat[var_name][0][0][8][0][ch_ind][0].T
        return num_channels,ch_names,topography_3D