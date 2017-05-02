from mne.io import read_raw_fif
from mne.channels import read_layout, find_layout,read_ch_connectivity
from mne.datasets import sample
import numpy as np
from mne.viz import plot_topomap
import os
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.io import loadmat

class Convolutions:
    #TODO write method for getting service data and override it in child classes
    def __init__(self,neighboring,topography_3D,ch_names,num_channels):
        self.neighboring = neighboring
        self.topography_3D = topography_3D
        self.ch_names = ch_names
        self.num_channels = num_channels
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

class ConvolutionsNeuromag(Convolutions):
    #
    def __init__(self,sensor_type='mag'):
        # @sensor_type "mag" or "grad"
        sensor_type = sensor_type
        code_dir = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(code_dir,'neuromag_info')
        if (sensor_type == 'mag'):
            neighboring_filename = os.path.join(info_path,'neuromag306mag_neighb.mat')
        if (sensor_type == 'grad'):
            neighboring_filename = os.path.join(info_path,'neuromag306planar_neighb.mat')
        neuromag = read_raw_fif(sample.data_path() +
                          '/MEG/sample/sample_audvis_raw.fif')
        self.topography_2D = find_layout(neuromag.info, ch_type=sensor_type).pos
        topography_3D = np.array([ch['loc'][:3] for ch in neuromag.info['chs'] if (ch['ch_name'][-1] == '1') & (ch['ch_name'][0:3] == 'MEG')])

        neighboring,ch_names = read_ch_connectivity(neighboring_filename, picks=None) #ch. names written  in 'MEG1111' format
        neighboring = neighboring.toarray()
        num_channels = len(ch_names)
        Convolutions.__init__(self, neighboring,topography_3D,ch_names,num_channels)

class ConvolutionsGSN128(Convolutions):
    def __init__(self):
        code_dir = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(code_dir,'gsn128_info')
        neighboring_filename = os.path.join(info_path,'gsn128_neighb.mat')
        self.topography_2D = read_layout('GSN-128.lay', info_path).pos[:,:2]
        num_channels,ch_names,topography_3D = self._parse_mat(os.path.join(info_path,'bs_topography.mat'))

        neighboring,self.ch_names = read_ch_connectivity(neighboring_filename, picks=None)
        neighboring = neighboring.toarray()
        Convolutions.__init__(self, neighboring, topography_3D, ch_names, num_channels)

    def _parse_mat(self,mat_file):
        #This function parses topogpraphy .mat from brainstorm. It's really weird magic
        #TODO rewrite search inside imported mat file using read_ch_connectivity mne code
        mat = loadmat(mat_file);
        var_name = mat.keys()[1]

        num_channels = len(mat[var_name][0][0][8][0])
        ch_names = []
        topography_3D = np.zeros((num_channels,3))
        for ch_ind in xrange(num_channels):
            ch_names.append(mat[var_name][0][0][8][0][ch_ind][5][0])
            topography_3D[ch_ind,:]= mat[var_name][0][0][8][0][ch_ind][0].T
        return num_channels,ch_names,topography_3D



#TODO rewrite this as child for Convolutions class
class VisualisationConvolutions:
    def __init__(self,convolutions_object):
        self.topography_2D=convolutions_object.topography_2D
        self.ch_names = convolutions_object.ch_names
        self.num_channels = convolutions_object.num_channels

    def _plot_custom_topography(self):
        plt.scatter(self.topography_2D[:,0], self.topography_2D[:,1], s=0.5)

    def _plot_convolution(self, sensors_indices, conv_score, radius=0.5, connect=False, draw_normal=False, plot_only_middle=False):
        # plot convolution primitive
        # @sensors_indices indices of channels, forming convolution
        # @conv_score - meajure of activation of convolution, have to be [0:1]
        cm = matplotlib.cm.get_cmap('gist_rainbow')
        color = cm(conv_score)
        conv_length = len(sensors_indices)
        is_even = lambda x: x % 2 == 0

        xs = [self.topography_2D[sensor_index][0] for sensor_index in sensors_indices]
        ys = [self.topography_2D[sensor_index][1] for sensor_index in sensors_indices]

        if plot_only_middle:
            if ~is_even(conv_length):
                plt.scatter(xs[conv_length/2], ys[conv_length/2], s=radius, c=color)
        else:
            plt.scatter(xs, ys, s=radius, c=color)

        if (connect) & ~(plot_only_middle):
            plt.plot(xs, ys, c=color)

        if draw_normal:
            vectors_xs = np.array([xs[i] - xs[i + 1] for i in range(conv_length - 1)])
            vectors_ys = np.array([ys[i] - ys[i + 1] for i in range(conv_length - 1)])
            norms = np.linalg.norm(np.vstack((vectors_xs, vectors_ys)), axis=0)
            vectors_xs /= norms
            vectors_ys /= norms
            conv_normal = np.array((sum(vectors_ys), -sum(vectors_xs)))  # Swap coordinate to get normal vector
            eps = 0.02 #constant to make visible normals, wich close to zero
            conv_normal = conv_score*0.05 * conv_normal + eps

            if ~is_even(conv_length):
                central_node = sensors_indices[conv_length // 2]
                start_x, start_y = self.topography_2D[central_node][:2]
                plt.plot([start_x - 0.5 * conv_normal[0], start_x + 0.5 * conv_normal[0]],
                         [start_y - 0.5 * conv_normal[1], start_y + 0.5 * conv_normal[1]], c=color)
            plt.gca().set_aspect('equal', adjustable='box')

    #TODO rewrite _visualise_convolution and _visualise_target_convolutions as one function with decorator
    def _visualise_convolution(self, convs):
        # This function draw one convolution on custom topography with direction of convolution

        test_conv = './results/test_conv'
        if not os.path.isdir(test_conv):
            os.makedirs(test_conv)
        for conv_index, conv in enumerate(convs):
            conv_names = [self.ch_names[index] for index in conv]
            title = '_'.join([name[3:] for name in conv_names])
            plt.title('%s,svd ratio' % (title))
            self._plot_custom_topography()
            conv_score=1.0
            self._plot_convolution(conv, conv_score, radius=5, connect=True, draw_normal=True)

            plt.savefig(os.path.join(test_conv, title + '.png'))
            plt.close()

    def _visualise_target_convolutions(self, convs,conv_scores,title):
        # This function draw all specified convolution on one custom topography with direction of convolution
        #
        resutls_dir = './resutls'
        if not os.path.isdir(resutls_dir):
            os.makedirs(resutls_dir)

        self._plot_custom_topography()
        for conv_index, conv in enumerate(convs):
            plt.title('%s,svd ratio' % (title))
            self._plot_convolution(conv, conv_scores[conv_index], radius=5, connect=False, draw_normal=True,plot_only_middle=True)

        plt.savefig(os.path.join(resutls_dir, title + '.png'))
        plt.close()

if __name__=='__main__':
    cn = ConvolutionsNeuromag()
    convs = cn.get_1Dconvolution_channels(3)
    vs = VisualisationConvolutions(cn)
    vs._visualise_target_convolutions(convs[1:10],np.random.rand(9),'qqwerty')
    
    import conv_trees as cc
    # test of path tracing (have no use just now)
    init_v = 0
    target_v = 90
    (result, path) = cc.find_path(cn.topography_3D,cc.curr_faces,init_v,target_v)
    if result:
        trace_seq = cc.trace_path(cn.topography_3D, cc.curr_faces, init_v, target_v, path)
        cc.plot_tracing_results(cn.topography_3D,cc.curr_faces,trace_seq)
    # test for convolutions --- TODO: read real convolutions
    test_convs = np.array([])
    res = cc.make_geodesic_conv_combinations(cn.topography_3D, test_convs, 3, 0.1, 0.1, 0.1, cc.curr_faces)
    #plot_combination(cn.topography_3D,cc.curr_faces,test_convs,res[0][0])
    
    print 'ok'


