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
        
        self.neighboring,self.ch_names = read_ch_connectivity(neighboring_filename, picks=None) #ch. names written  in 'MEG1111' format
        self.neighboring = self.neighboring.toarray()
        self.num_channels = len(self.ch_names)
        self.edges_matrix = self.calc_edges_matrix()
        #self.neighboring = neighboring.toarray()

    # def visualise_sensors(self):

    def calc_edges_matrix(self):
        #We need only upper part of connectivity matrix
        pos = self.layout.pos[:, 0:2]
        neib_array = self.neighboring
        edges_matrix = np.empty(self.neighboring.shape,dtype=object)
        edges_matrix.fill(np.array([np.inf,np.inf]))
        for x in xrange(len(neib_array)):
            for y in xrange(x+1,len(neib_array)):
              if neib_array[x,y]:
                edge = np.array((pos[x][0]-pos[y][0],pos[x][1]-pos[y][1]))
                edges_matrix[x,y] = edge
        edges_matrix = edges_matrix.T + edges_matrix
        return edges_matrix


    def is_1D_planar(self, convolution,threshold=2.5):
        #TODO Threshold havily depends from conv length. Research this dependecy
        conv_length = len(convolution)
        vectors = []
        for vertex_index in range(conv_length - 1):
            start_index = self.ch_names.index(convolution[vertex_index])
            end_index = self.ch_names.index(convolution[vertex_index + 1])
            if self.edges_matrix[start_index][end_index]:
                vectors.append(np.array(self.edges_matrix[start_index][end_index]))
            else:
                print 'Some goes wrong'
        vectors_mat = np.array(vectors)
        _, s, _ = np.linalg.svd(vectors_mat)
        return (s[0] / s[1:].sum()) > threshold

    def __empirical_threshold_calculation__(self):
        conv = ['MEG0141', 'MEG1541', 'MEG1531', 'MEG1711'] #presumably worst case for 2d head projection
        res = cn.is_1D_planar(conv)

        print res

    def get_1Dconvolutions(self,conv_length):
        # def recursive_search(curr_ch,conv_length,avail_channels,neighboring):
        #     #@curr_ch - string, name of start channel in 1D convolution
        #     #@returned list of lists-potential convolutions
        #
        #     res = []
        #     np.fill_diagonal(neighboring, 0)
        #     curr_ch_index=avail_channels.index(curr_ch)
        #     if conv_length == 1:
        #         for index,potent_neigh in enumerate(avail_channels):
        #             if neighboring[curr_ch_index][index]:
        #                 res.append([potent_neigh])
        #         return res
        #     # edge_matrix[avail_channels.index(curr_ch)]
        #     new_avail_channels = [ch for ch in avail_channels if ch != curr_ch]
        #     new_neighboring = np.delete(neighboring, curr_ch_index, 0)
        #     new_neighboring = np.delete(new_neighboring, curr_ch_index, 1)
        #     for index,potent_neigh in enumerate(avail_channels):
        #         if neighboring[curr_ch_index][index]:
        #             accum = recursive_search(potent_neigh, conv_length-1, new_avail_channels, new_neighboring)
        #             accum = [[potent_neigh]+seq for seq in accum]
        #             res = res+accum
        #             return res

        def recursive_search(curr_ch_index, potential_neighbors, conv_length):
            result = []
            if conv_length == 1:
                for p_n_index in potential_neighbors:
                    if self.neighboring[curr_ch_index][p_n_index]:
                        result.append([p_n_index])
                # return result
            else:
                for p_n_index in potential_neighbors:
                    if self.neighboring[curr_ch_index][p_n_index]:
                        new_potential_neighbors = [ch for ch in potential_neighbors if ch != p_n_index]
                        tmp_res = recursive_search(p_n_index,new_potential_neighbors,conv_length-1)
                        result = result + tmp_res
            result = [[curr_ch_index] + elem for elem in result]
            return result
        res = []
        for ch_index in range(self.num_channels):
            res.append(ch_index, range(1, len(self.ch_names)), conv_length)

        return res


if __name__=='__main__':
    cn = convolution_neuromag('mag')
    res = cn.get_1Dconvolutions(2)
    res1 = [[cn.ch_names[index] for index in elem] for elem in res]
    print res


