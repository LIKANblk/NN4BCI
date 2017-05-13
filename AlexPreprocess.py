
from scipy.linalg import norm
from scipy.signal import hilbert
from itertools import combinations
from Data import *
from convolutions import *

#def get_convovled_data(data,convolutions_1d):
#    for conv in convolutions_1d:
#        yield data[:,conv,:]


class AlexPreprocess:
    #TODO tune beta parameter
    def __init__(self):
        pass

    
    def extract_expert_features(self,data,convolution_indeces,ch_dim,beta=1):
        #TODO too slow, make parallel feature computations
        assert(data.ndim == 3)
        time_dim = data.ndim - ch_dim
        phase_data = self._calc_local_phase(data,time_dim)
        res_shape = list(data.shape)
        res_shape[ch_dim] = len(convolution_indeces)
        res = np.zeros(res_shape)
        for index, data_chunk4conv in enumerate(self._get_convolved_data(phase_data,convolution_indeces,ch_dim)):
            indth = [slice(None)] * ch_dim + [index] + [slice(None)] * (data.ndim - ch_dim - 1)
            res[indth]=self._get_Alex_feature(data_chunk4conv, beta, feat_dim=ch_dim)
        return res



    def _calc_local_phase(self,data,time_dim):    
        analytic_signal = hilbert(data,axis = time_dim)
        return np.unwrap(np.angle(analytic_signal))

    def _get_convolved_data(self,data,convolutions_1d,ch_dim):
        for convolution in convolutions_1d:
            ind = [slice(None)] * ch_dim + [convolution] + [slice(None)] * (data.ndim - ch_dim - 1)
            yield data[ind]
    
    def _get_Alex_feature(self,conv_phase_values,beta,feat_dim):
        conv_length = conv_phase_values.shape[feat_dim]
        comb = list(combinations(range(conv_length), 2))
        out_shape = list(conv_phase_values.shape)
        out_shape[feat_dim]=len(comb)
        conv_phase_delta = np.zeros(out_shape)
        for ind,(i,j) in enumerate(comb):
            ith = [slice(None)] * feat_dim + [i] + [slice(None)] * (conv_phase_values.ndim - feat_dim - 1)
            jth = [slice(None)] * feat_dim + [j] + [slice(None)] * (conv_phase_values.ndim - feat_dim - 1)
            indth = [slice(None)] * feat_dim + [ind] + [slice(None)] * (conv_phase_values.ndim - feat_dim - 1)
            conv_phase_delta[indth] = conv_phase_values[ith]-conv_phase_values[jth]
        return np.exp(-beta * norm(conv_phase_delta,axis = feat_dim) ** 2)


    def visualise_activations(self,vis_object,extracted_features,convs_indices):
        #TODO do smth with conv indices
        # color_normaliser = matplotlib.colors.Normalize(vmin=extracted_features.min(), vmax=extracted_features.min())

        for t in xrange(extracted_features.shape[1]):
            title = 'Time %f' %t
            vis_object._visualise_target_convolutions(convs_indices,extracted_features[0,:,t], title) #ATTENSION temporary constant



if __name__ == '__main__':
    data_source = NeuromagData('mag')
    X, _ = data_source.get_data_from_exp('em01',target_dim_order=['trial', 'time', 'channel'])
    dev = Neuromag('mag')
    cn = Convolutions(dev)
    convs = cn.get_1Dconvolutions(3)
    # ap = AlexPreprocess()
    features = AlexPreprocess().extract_expert_features(X,convs,ch_dim=2)
    vc = VisualisationConvolutions(cn)
    # ap.visualise_activations(vc,features,convs)
    print 'ok'
