# import mne
from mne.io import read_raw_fif
from mne.channels import find_layout,read_ch_connectivity
from mne.datasets import sample
from scipy.sparse import find

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
        #self.neighboring = neighboring.toarray()

    # def visualise_sensors(self):

    def cacl_edges(self):
        #We need only upper part of connectivity matrix
        pos = self.layout.pos[:, 0:1]
        res=[]
        neib_array = self.neighboring.toarray()
        for x in xrange(len(neib_array)):
            for y in xrange(x,len(neib_array)):
              if neib_array[x,y]:
                edge = (pos[x][0]-pos[y][0],pos[x][1]-pos[y][1])
                res.extend((self.ch_names[x],self.ch_names[y],edge))


    def get_1Dconvolutions(self,conv_length=2):
        pos = self.layout.pos[:,0:1] #get only (x,y) coordinates (without sensors width and height)
        num_channels = pos.shape[0]
        index_neighboring = find(self.neighboring) #tuple of non-zero elems
        for ch in xrange(num_channels):
            for





