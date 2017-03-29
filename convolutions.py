from mne.io import read_raw_fif
from mne.channels import find_layout,read_ch_connectivity
from mne.datasets import sample
import numpy as np
from mne.viz import plot_topomap
import os
import matplotlib.pyplot as plt


class ConvolutionNeuromag:
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
        self.topography_2D = find_layout(neuromag.info, ch_type=self.sensor_type).pos
        self.topography_3D = np.array([ch['loc'][:3] for ch in neuromag.info['chs'] if (ch['ch_name'][-1] == '1') & (ch['ch_name'][0:3] == 'MEG')])

        self.neighboring,self.ch_names = read_ch_connectivity(neighboring_filename, picks=None) #ch. names written  in 'MEG1111' format
        self.neighboring = self.neighboring.toarray()
        self.num_channels = len(self.ch_names)
        self.edges_matrix = self.calc_3D_edges_matrix()


    def calc_3D_edges_matrix(self):
        #We need only upper part of connectivity matrix
        neib_array = self.neighboring
        edges_matrix = np.empty(self.neighboring.shape,dtype=object)
        edges_matrix.fill(np.array([np.inf,np.inf]))
        for x in xrange(len(neib_array)):
            for y in xrange(0,len(neib_array)):
              if neib_array[x,y]:
                edge = np.array((self.topography_3D[x][0]-self.topography_3D[y][0],self.topography_3D[x][1]-self.topography_3D[y][1],
                                 self.topography_3D[x][2]-self.topography_3D[y][2]))
                edges_matrix[x,y] = edge
        return edges_matrix


    def is_1D_planar(self, convolution,threshold=0.01):
        #@convolution - short list of channel indeceses to be convolved
        # @threshold - ration between singular values, larger wich we consider convolution 1D_planar
        #TODO improve estimation of matrix rank
        vectors_mat = np.array(self.topography_3D[convolution])
        # _,s,_ = np.linalg.svd(vectors_mat)
        return np.linalg.matrix_rank(vectors_mat,threshold) < 3 #TODO DEBUG threshold

    def __empirical_threshold_calculation__(self):
        conv = ['MEG0221', 'MEG0441', 'MEG1821', 'MEG1831'] #presumably worst case for 2d head projection
        conv_index = [self.ch_names.index(name) for name in conv]
        res = cn.is_1D_planar(conv_index)
        print res

    def get_1Dconvolution_channels(self,conv_length):
        # Method for searching 1D planara subgraphs
        def recursive_search(curr_ch_index, potential_neighbors, conv_length):
            result = []
            if conv_length == 2: #TODO HACK please fix this
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
            print ch_index
            potential_neighbors = (range(len(self.ch_names)))
            potential_neighbors.remove(ch_index)
            res.extend(recursive_search(ch_index, potential_neighbors, conv_length))
        result = filter(lambda x:self.is_1D_planar(x),res)
        return result

    def _visualise_convolutions(self,convs):
        #This function used to test correctness of finded convolutions
        test_conv = './test_conv'
        if not os.path.isdir(test_conv):
            os.makedirs(test_conv)
        for conv_index, conv in enumerate(convs):
            fake_data = np.zeros(self.num_channels)
            fake_data[conv] = 100
            conv_names = [self.ch_names[index] for index in conv]
            title = '_'.join([name[3:] for name in conv_names])
            plt.title('%s,svd ratio' % (title))
            im,_ = plot_topomap(data=fake_data, pos=self.topography_2D,contours=0,names=self.ch_names,show=False)
            plt.savefig(os.path.join(test_conv,title+'.png'))
            plt.close()

    def _plot_custom_topography(self,sensors_indices,color='b',radius=0.5,connect=False,draw_normal=False):

        # fig, ax = plt.subplots(1, 1)
        xs = [self.topography_2D[sensor_index][0] for sensor_index in sensors_indices]
        ys = [self.topography_2D[sensor_index][1] for sensor_index in sensors_indices]

        plt.scatter(xs, ys, s=radius, c=color)
        if False:
            plt.plot(xs,ys,c=color)

        is_even = lambda x: x % 2 == 0
        conv_length = len(sensors_indices)
        if draw_normal:
            vectors_xs = np.array([xs[i] - xs[i+1] for i in range(conv_length - 1)])
            vectors_ys = np.array([ys[i] - ys[i+1] for i in range(conv_length - 1)])
            norms = np.linalg.norm(np.vstack((vectors_xs,vectors_ys)),axis=0)
            vectors_xs /= norms
            vectors_ys /= norms
            conv_normal = np.array((sum(vectors_ys),-sum(vectors_xs))) #Swap coordinate to get normal vector
            conv_normal = 0.1*conv_normal/np.linalg.norm(conv_normal)


            if ~is_even(conv_length):
                central_node = sensors_indices[conv_length//2]
                start_x,start_y=self.topography_2D[central_node][:2]
                plt.plot([start_x-0.5*conv_normal[0], start_x + 0.5*conv_normal[0]],
                         [start_y- 0.5*conv_normal[1], start_y + 0.5*conv_normal[1]],c=color)
            plt.gca().set_aspect('equal', adjustable='box')

    def _visualise_directions(self,convs):
        #This function visualise each convolution on custom topography with direction of convolution

        test_conv = './test_conv1'
        if not os.path.isdir(test_conv):
            os.makedirs(test_conv)
        for conv_index, conv in enumerate(convs):
            conv_names = [self.ch_names[index] for index in conv]
            title = '_'.join([name[3:] for name in conv_names])
            plt.title('%s,svd ratio' % (title))
            self._plot_custom_topography(range(self.num_channels))
            self._plot_custom_topography(conv,color='r',radius=5,connect=True,draw_normal=True)

            plt.savefig(os.path.join(test_conv,title+'.png'))
            plt.close()

if __name__=='__main__':
    cn = ConvolutionNeuromag('mag')
    convs = cn.get_1Dconvolution_channels(3)
    cn._visualise_directions(convs)

