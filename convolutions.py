from mne.viz import plot_topomap
import matplotlib.pyplot as plt
import matplotlib.cm
from itertools import product
import math
from devices import *



class Convolutions:
    #TODO write method for getting service data and override it in child classes
    def __init__(self,device):
        self.neighboring = device.neighboring
        self.topography_2D = device.topography_2D
        self.topography_3D = device.topography_3D
        self.ch_names = device.ch_names
        self.num_channels = device.num_channels
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




    def get_1Dconvolution_old(self, conv_length):
        # TODO rewrite this pice of shit
        # DEPRICATED! Method for searching 1D planar subgraphs

        def is_1D_planar(self, convolution, eps=0.01):
            # @convolution - short list of channel indeceses to be convolved
            # @threshold - ration between singular values, larger wich we consider convolution 1D_planar
            # TODO improve estimation of matrix rank
            vectors_mat = np.array(self.topography_3D[convolution])
            # return np.linalg.matrix_rank(vectors_mat,threshold) < 3 #TODO DEBUG threshold
            _, s, _ = np.linalg.svd(vectors_mat)
            threshold = s.max() * max(vectors_mat.shape) * eps
            return sum(s > threshold) < 3

        def remove_duplicates(convs):
            res = []
            for conv in convs:
                s_conv = sorted(conv)
                if all(map(lambda x: sorted(x) != s_conv, res)):
                    res.append(conv)
            return res

        def recursive_search(curr_ch_index, potential_neighbors, conv_length):
            result = []
            if conv_length == 2:  # TODO HACK please fix this
                for p_n_index in potential_neighbors:
                    if self.neighboring[curr_ch_index][p_n_index]:
                        result.append([p_n_index])
                        # return result
            else:
                for p_n_index in potential_neighbors:
                    if self.neighboring[curr_ch_index][p_n_index]:
                        new_potential_neighbors = [ch for ch in potential_neighbors if ch != p_n_index]
                        tmp_res = recursive_search(p_n_index, new_potential_neighbors, conv_length - 1)
                        result = result + tmp_res
            result = [[curr_ch_index] + elem for elem in result]
            return result

        res = []
        for ch_index in range(self.num_channels):
            # print ch_index
            potential_neighbors = (range(len(self.ch_names)))
            potential_neighbors.remove(ch_index)
            res.extend(recursive_search(ch_index, potential_neighbors, conv_length))

        result = filter(lambda x: is_1D_planar(x), res)
        result = remove_duplicates(result)
        return result


    def _angle_betwen_edges(self,edge1,edge2):
        #Calculate angel between edges in graph
        # edge  - tuple with two vertices (start of the edge end end of the edge)
        vector1 = self.topography_3D[edge1[1]] - self.topography_3D[edge1[0]]
        vector2 = self.topography_3D[edge2[1]] - self.topography_3D[edge2[0]]
        return math.acos(vector1.dot(vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))

    def get_1Dconvolutions(self,conv_length,thres = (math.pi/6)):
        def generate_pairs():
            #generate pairs of neighboring channels (start for further concatenation
            res=[]
            for ch_y in xrange(self.num_channels):
                for ch_x in xrange(ch_y+1,self.num_channels):
                    if self.neighboring[ch_y][ch_x]:
                        res.append([ch_y,ch_x])
            return res
        def is_planar(conv):
            # Check is conv is plannar
            # We checking only  right most and left most pairs of convolution edges
            #@conv - list of vertices
            left_is_planar = (self._angle_betwen_edges((conv[0],conv[1]), (conv[1],conv[2])) <(thres))
            right_is_planar = (self._angle_betwen_edges((conv[0], conv[1]), (conv[1], conv[2])) < (thres))
            return left_is_planar & right_is_planar

        def union_lists(l1,l2):
            #Union  lists l1 and l2 if they have apropriate intersection
            f= lambda l: l if is_planar(l) else []
            if (l1[1:]==l2[0:-1]):
                return f(l1 +[l2[-1]])
            if (l1[0:-1] == l2[1:]):
                return f(l2 + [l1[-1]])
            if (l1[1:] == l2[-1:0:-1]):
                return f(l1 + [l2[0]])
            if (l1[0:-1] == l2[-2::-1]):
                return f([l2[-1]] + l1)
            return []


        def conv_concat(convs):
            #Concatenate (union) neighboring convolutions
            res = []
            for i in xrange(len(convs)):
                for j in xrange(i+1,len(convs)):
                    intersect_res = union_lists(convs[i],convs[j])
                    if len(intersect_res)>0:
                        res.append(intersect_res)
            return res

        convs = generate_pairs()
        while len(convs[0]) < conv_length:
            convs = conv_concat(convs)
        return convs

    def _test_convs_correctness_(self,convs):
        #Test method for estimating corretness of array of convs
        def remove_duplicates(convs):
            res = []
            for conv in convs:
                s_conv = sorted(conv)
                if all(map(lambda x: sorted(x) != s_conv, res)):
                    res.append(conv)
            return res

        #search vertex duplicates inside each conv
        conv_length = len(convs[0])
        no_loops = all([len(set(elem))==conv_length for elem in convs])
        if not no_loops:
            print 'Some convolutions have duplicate vertices'

        # search conv. duplicates inside convs list
        no_duplicates = len(remove_duplicates([sorted(elem) for elem in convs])) == len(convs)
        if not no_duplicates:
            print 'Some convolutions met several times in convs list'

    def get_crosses_conv(self,conv_length):
        #get cross-like convolutions
        #@return array of tuples, where each tuple consistc of two lists, each list 1d-planar convolution
        convs = self.get_1Dconvolutions(conv_length)
        res = []
        if conv_length%2 == 1:
            middle_vertex_ind = int(np.ceil(conv_length/2))
            for conv_index, conv in enumerate(convs):
                for p_c_ind,p_c in enumerate(convs[conv_index+1:]):
                    if ((p_c[middle_vertex_ind]==conv[middle_vertex_ind]) & (conv_index != p_c_ind)):
                        neighb_mask = [self.neighboring[conv[i],p_c[j]] for i,j in product([middle_vertex_ind+1,middle_vertex_ind-1],repeat=2)]
                        if all(neighb_mask):
                            res.append((conv,p_c))
        return res

    def get_conv_vectors(self,conv):
        # Function for calculating tangent and normal vector of convolution
        conv_length = len(conv)
        xs = [self.topography_3D[sensor_index][0] for sensor_index in conv]
        ys = [self.topography_3D[sensor_index][1] for sensor_index in conv]
        vectors = [np.array((xs[i] - xs[i + 1],ys[i] - ys[i + 1])) for i in range(conv_length - 1)]
        normalised_vectors = map(lambda x: x/np.linalg.norm(x),vectors)
        tangent = sum(normalised_vectors)/np.linalg.norm(sum(normalised_vectors))
        normal = np.array((tangent[1],-tangent[0]))
        return tangent,normal


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
        resutls_dir = './results'
        if not os.path.isdir(resutls_dir):
            os.makedirs(resutls_dir)

        self._plot_custom_topography()
        for conv_index, conv in enumerate(convs):
            plt.title('%s,svd ratio' % (title))
            self._plot_convolution(conv, conv_scores[conv_index], radius=5, connect=True, draw_normal=False,plot_only_middle=False)

        plt.savefig(os.path.join(resutls_dir, title + '.png'))
        plt.close()

    def visualise_convs_on_mne_topomap(self, convs):
        # This function used to test correctness of finded convolutions
        test_conv = './test_conv'
        if not os.path.isdir(test_conv):
            os.makedirs(test_conv)
        for conv_index, conv in enumerate(convs):
            fake_data = np.zeros(self.num_channels)
            fake_data[conv] = 100
            conv_names = [self.ch_names[index] for index in conv]
            title = '_'.join([name for name in conv_names])
            plt.title('%s' % (title))
            im, _ = plot_topomap(data=fake_data, pos=self.topography_2D, contours=0, names=self.ch_names,
                                 show=False)
            plt.savefig(os.path.join(test_conv, title + '.png'))
            plt.close()

def Peters_test(convs):
    import conv_trees as cc
    # test of path tracing (have no use just now)
    init_v = 0
    target_v = 90
    (result, path) = cc.find_path(cn.topography_3D,cc.curr_faces,init_v,target_v)
    if result:
        trace_seq = cc.trace_path(cn.topography_3D, cc.curr_faces, init_v, target_v, path)
        cc.plot_tracing_results(cn.topography_3D,cc.curr_faces,trace_seq)
    # test for convolutions --- TODO: read real convolutions
    test_convs = np.array(convs)
    res = cc.make_geodesic_conv_combinations(cn.topography_3D, test_convs, 3, 0.1, 0.1, 0.1, cc.curr_faces, 'directions_Real.csv')
    #plot_combination(cn.topography_3D,cc.curr_faces,test_convs,res[0][0])
    return res

def cross_convs_vis_test():
    dev = Neuromag('mag')
    cn = Convolutions(dev)
    vs = VisualisationConvolutions(cn)
    cross_convs = cn.get_crosses_conv(3)
    vs.visualise_convs_on_mne_topomap(map(lambda inp_tuple:inp_tuple[0]+inp_tuple[1],cross_convs))

def convs_vis_test():
    dev = Neuromag('mag')
    cn = Convolutions(dev)
    vs = VisualisationConvolutions(cn)
    convs = cn.get_1Dconvolutions(3)
    vs.visualise_convs_on_mne_topomap(convs)

if __name__=='__main__':
    dev = Neuromag('mag')
    cn = Convolutions(dev)
    convs = cn.get_1Dconvolutions(3)
    import conv_trees as cc
    res = cc.make_geodesic_conv_combinations(dev.topography_3D, np.array(convs), 3, 0.4, 0.4, 0.3, cc.curr_faces,
                                             'directions_Real.csv')
    res = filter(lambda x: x[1]<0.15,res)
    vs = VisualisationConvolutions(cn)
    for ind,elem in enumerate(res):
        convs_inds = elem[0]
        tmp_convs = [convs[i] for i in convs_inds]
        # vs.visualise_convs_on_mne_topomap(tmp_convs,'%s' % ind)
        vs._visualise_target_convolutions(tmp_convs, np.ones(len(convs)), '%s' % ind)
    print 'ok'



