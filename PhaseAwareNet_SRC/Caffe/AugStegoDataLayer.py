# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from random import shuffle
#from PIL import Image

import matplotlib.image as mpimg


class AugmentDataLayerSync(caffe.Layer):

    """
    This is a simple syncronous datalayer for inputting the augmented data layer on the fly
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the paramameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader( params, None )

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape( self.batch_size, 
                        1, 
                        params['im_shape'][0], 
                        params['im_shape'][1] )
        
        # Ground truth
        top[1].reshape(self.batch_size)

        print_info( "AugmentStegoDataLayerSync", params )

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, 0, :, :] = im
            top[1].data[itt] = label

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        
        self.result = result
        self.batch_size = params['batch_size']
        self.root = params['root']
        self.im_shape = params['im_shape']
        self.trainMode = ( params['split'] == 'train' )  # determine the mode, if test, no augment        
         
        # get list of image indexes.
        list_file = params['split']  + '.txt'
        TXT_FILE = osp.join( self.root, list_file )
        txt_lines = [ line.rstrip('\n') for line in open( TXT_FILE ) ]
    
        total_size = len( txt_lines )
        
        assert total_size%2 == 0,  "total_size must be even" 
    
        self.images = []
        self.labels = np.zeros( ( total_size, ), dtype = np.int64 )
        self.indexlist = range( total_size )

        for i in np.arange(total_size):
            tmp = txt_lines[i].split()
            self.images.append(tmp[0])
            self.labels[i] = int(tmp[1])
                
        self._cur = 0  # current image
        self._epoch = 0  # current epoch count, also used as the randomization seed
        self._flp = 1  # Augment flip number,
        self._rot = 0  # Augment rotation number
     
        print "BatchLoader initialized with {} images".format(len(self.indexlist))

    def load_next_image( self ):
        
        """
        Load the next image in a batch
        """
        # Did we finish an epoch
        if self._cur == len(self.indexlist):
            self._epoch += 1
            l = np.random.seed( self._epoch )  #randomize, aslo reproducible
            l = np.random.permutation( len(self.indexlist)/2 )
            l2 = np.vstack( ( 2*l, 2*l + 1 )).T
            self.indexlist = l2.reshape(len(self.indexlist),)
            self._cur = 0
            
        # Index list
        index = self.indexlist[self._cur]
        
        #load an image
        image_file_name = self.images[index]
        
        im = np.asarray( mpimg.imread( image_file_name ))
    
        #Determine the new fliplr and rot90 status, used it in the stego            
        if ( self.trainMode ):
            if ( self._cur % 2 == 0 ):
                self._flp = np.random.choice(2)*2 - 1
                self._rot = np.random.randint(4)
            im = im[:,::self._flp]
            im = np.rot90(im, self._rot)

        #load the ground truth
        label = self.labels[index]

        self._cur += 1
    
        return im, label
       

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])
    
    
