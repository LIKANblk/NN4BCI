import eeglearn.eeg_cnn_lib as eeglib
from devices import *
from Data import *
import matplotlib.pyplot as plt

if __name__ =='__main__':
    dev = Neuromag('mag')
    data_source = NeuromagData('mag')
    X,y = data_source.get_data_from_exp('em01',target_dim_order=['trial', 'time', 'channel'])
    images = eeglib.gen_images(locs=dev.topography_2D,features=X[0],n_gridpoints=200)
    images_path = 'results'
    for ind in xrange(images.shape[0]):
        plt.imshow(images[ind,0,:,:])
        plt.savefig(os.path.join(images_path,'%d.png' % ind))
        plt.close()
    print 'ok'
