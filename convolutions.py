# import mne
from mne.io import read_raw_fif
from mne.channels import find_layout,read_ch_connectivity
from mne.datasets import sample
from scipy.sparse import find
import numpy as np

class convolution_neuromag:
    #
    def __init__(self,sensor_type):
        # @sensor_type "mag" or "grad"
        self.sensor_type = sensor_type
        if (self.sensor_type == 'mag'):
            neighboring_filename = 'neuromag306mag_neighb.mat'
        if (self.sensor_type == 'grad'):
            neighboring_filename = 'neuromag306planar_neighb.mat'
        neuromag = read_raw_fif(sample.data_path() +
                          '/MEG/sample/sample_audvis_raw.fif')
        self.layout = find_layout(neuromag.info, ch_type=self.sensor_type)
        
        self.neighboring,self.ch_names = read_ch_connectivity(neighboring_filename, picks=None)
        self.num_channels = len(self.ch_names)
        #self.neighboring = neighboring.toarray()

    # def visualise_sensors(self):

    def cacl_edges_matrix(self):
        #We need only upper part of connectivity matrix
        pos = self.layout.pos[:, 0:1]
        neib_array = self.neighboring.toarray()
        edges_matrix = np.empty(self.neighboring.shape,dtype=object)
        for x in xrange(len(neib_array)):
            for y in xrange(x+1,len(neib_array)):
              if neib_array[x,y]:
                edge = (pos[x][0]-pos[y][0],pos[x][1]-pos[y][1])
                edges_matrix[x,y]=edge





    def get_1Dconvolutions(self,conv_length):

        def _find_next_neighbor(self, row_of_edge_matrix, avail_channels):
            # TODO Rewrite as lambda expression
            for ch_number in xrange(row_of_edge_matrix):
                if (row_of_edge_matrix[ch_number]):
                    return avail_channels[ch_number]
        def recursive_search(curr_ch,conv_length,avail_channels,edge_matrix):
            res = []
            curr_ch_index=avail_channels.index(curr_ch)
            if conv_length == 1:
                for index,potent_neigh in enumerate(avail_channels):
                    if edge_matrix[curr_ch_index][index]:
                        res.extend([potent_neigh])
                return res
                edge_matrix[avail_channels.index(curr_ch)]
            for index,potent_neigh in enumerate(avail_channels):
                if edge_matrix[curr_ch_index][index]:
                    new_avail_channels = [ch for ch in avail_channels if ch != potent_neigh]

                    new_edge_matrix = np.delete(edge_matrix, index, 0)
                    new_edge_matrix = np.delete(edge_matrix, index, 1)

                    accum = recursive_search(potent_neigh, conv_length-1, new_avail_channels, new_edge_matrix)
                    accum = [potent_neigh+seq for seq in accum]
                    res = res+accum
                    return res



        edge_matrix = self.cacl_edges_matrix()
        recursive_search([], conv_length, self.ch_names, edge_matrix)



