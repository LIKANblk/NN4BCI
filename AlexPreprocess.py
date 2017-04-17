
from scipy.linalg import norm
from scipy.signal import hilbert
from itertools import combinations
from Data import *
from convolutions import *

def get_convovled_data(data,convolutions_1d):
    for conv in convolutions_1d:
        yield data[:,conv,:]


class AlexPreprocess:
    #TODO tune beta parameter
    def __init__(self):
        pass

    def extract_expert_features(self,data,convolution_indeces):
        #TODO too slow, make parallel feature computations
        phase_data = self._calc_local_phase(data)
        beta = 1
        trials,features_length,time_length = data.shape[0],len(convolution_indeces),data.shape[-1]
        res = np.zeros((trials,features_length,time_length))
        for index, data_chunk4conv in enumerate(get_convovled_data(phase_data,convolution_indeces)):
            res[:,index,:]=self._get_Alex_feature(data_chunk4conv, beta)
        return res



    def _calc_local_phase(self,data):            #TODO rewrite as generators for large amount of data
        analytic_signal = hilbert(data)
        return np.unwrap(np.angle(analytic_signal))


    def _get_Alex_feature(self,conv_phase_values,beta):
        conv_length = conv_phase_values.shape[-2]
        comb = list(combinations(range(conv_length), 2))
        conv_phase_delta = np.zeros((conv_phase_values.shape[0],len(comb),conv_phase_values.shape[2]))
        for ind,(i,j) in enumerate(comb):
            conv_phase_delta[:,ind,:] = conv_phase_values[:,i,:]-conv_phase_values[:,j,:]
        return np.exp(-beta * norm(conv_phase_delta,axis = 1) ** 2)


    def visualise_activations(self,vis_object,extracted_features,convs_indices):
        #TODO do smth with conv indices
        # color_normaliser = matplotlib.colors.Normalize(vmin=extracted_features.min(), vmax=extracted_features.min())

        for t in xrange(extracted_features.shape[1]):
            title = 'Time %f' %t
            vis_object._visualise_target_convolutions(convs_indices,extracted_features[0,:,t], title) #ATTENSION temporary constant



if __name__ == '__main__':
    synth_data = GSN128Data()
    data = synth_data.load_all_data()[[0],:,:]
    cn = ConvolutionsGSN128()
    convs = cn.get_1Dconvolution_channels(3)
    ap = AlexPreprocess()
    features = AlexPreprocess().extract_expert_features(data,convs)
    vc = VisualisationConvolutions(cn)
    ap.visualise_activations(vc,features,convs)
    print 'ok'