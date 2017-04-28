from Data import *
from convolutions import Convolutions
from devices import Neuromag

class CrossFeatures:
    def __init__(self,device):
        self.device = device
        self.convolutions = Convolutions(device)

    def _calc_variance(self,cross_conv_val,cross_conv):
        def _projection(variance_val, cross_conv):
            # This function projects second part of convolution on orthogonal (to frist part)
            # direction
            # @variance (trial x time) array of second 1d_convolution
            # @cross_conv - current cross-convolution with indexes of used channels
            _, normal = self.convolutions.get_conv_vectors(cross_conv[0])
            tangent, _ = self.convolutions.get_conv_vectors(cross_conv[1])
            variance_directed = variance_val[:,:, np.newaxis] * tangent[np.newaxis, :]
            return np.tensordot(variance_directed,normal,axes=([2,0]))

        var1 = np.std(cross_conv_val[0],axis=1)**2
        var2 = np.std(cross_conv_val[1], axis=1)**2
        #We need to normalise variance along first part of covolutions and variance along second part of convolution
        #to make them lay along orthogonal directions
        var2_normed = _projection(var2,cross_conv)
        return var1,var2_normed



    def _get_convolved_data(self,data,cross_convs,ch_dim):
        for cros_conv in cross_convs:
            horis_ind = [slice(None)] * ch_dim + [cros_conv[0]] + [slice(None)] * (data.ndim - ch_dim - 1)
            vert_ind = [slice(None)] * ch_dim + [cros_conv[1]] + [slice(None)] * (data.ndim - ch_dim - 1)
            yield (data[horis_ind],data[vert_ind])

    def extract_expert_features(self,data,conv_length=3):
        # @data - data array of kind trials x channels x time
        cross_convs_inds = self.convolutions.get_crosses_conv(conv_length)
        trial_dim = 0 #by default different trials by first(0) dim
        ch_dim = data.shape.index(self.device.num_channels)
        time_dim = [i for i in range(3) if i not in [trial_dim,ch_dim]]

        feat_num = len(cross_convs_inds)
        feat_dim = 2 #dimesionality of the space

        res_shape=list(data.shape)
        res_shape[ch_dim]=feat_num
        res_shape.insert(ch_dim+1,feat_dim)
        features = np.zeros((trials,feat_num,feat_dim,time_samples))
        for index, data_chunk4cr_conv in enumerate(self._get_convolved_data(data,cross_convs_inds)):
            iter_var1,iter_var2 = self._calc_variance(data_chunk4cr_conv,cross_convs_inds[index])
            features[:,index,:,:] = np.concatenate((iter_var1[:,np.newaxis,:],iter_var2[:,np.newaxis,:]),axis=1)
        return features

if __name__ == '__main__':
    dev = Neuromag('mag')
    convolutions_object = Convolutions(dev)
    cf = CrossFeatures(convolutions_object)
    data = NeuromagData('mag')
    data = data.get_data_by_label('em01',data.get_data_labels()[0])
    features = cf.extract_expert_features(data,conv_length=3)
    print features.shape


