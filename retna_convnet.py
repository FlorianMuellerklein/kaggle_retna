import os
import time
import theano
import lasagne
import skimage
import numpy as np
import pandas as pd
import cPickle as pickle
from random import randint, uniform

from skimage import exposure
from matplotlib import pyplot
from skimage.io import imread
from skimage.io import imshow
from skimage import transform

from lasagne import layers
from lasagne.updates import sgd
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

learning_rate = 0.003
PATH = 'train_grey.npy'
yPATH ='train_y.npy'
TESTPATH = 'test.npy'
PIXELS = 512
imageSize = PIXELS * PIXELS
num_features = imageSize
print num_features

def plot_sample(x):
    img = x.reshape(PIXELS, PIXELS)
    imshow(img)
    pyplot.show()

def load2d(PATH, yPATH, test):
    if test:
        images = np.load(PATH) #/ 255.
        y = np.ones(images.shape[0])
        images = images.astype(np.float32)
        images = images.reshape(-1, 1, PIXELS, PIXELS)
    else:
        images = np.load(PATH) #/255.
        y = np.load(yPATH)
        images = images.astype(np.float32)
        images = images.reshape(-1, 1, PIXELS, PIXELS)
    
    return images, y

def float32(k):
    return np.cast['float32'](k)
    
def plot_loss(net):
    """
    Plot the training loss and validation loss versus epoch iterations with respect to 
    a trained neural network.
    """
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.plot(train_loss, linewidth = 3, label = "train")
    pyplot.plot(valid_loss, linewidth = 3, label = "valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.yscale("log")
    pyplot.show()
    
        
class AdjustVariable(object):
    def __init__(self, name, start = learning_rate, stop = 0.0001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience = 100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()
            
class FlipBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]
        
        return Xb, yb
        
class DataAugmentationBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(DataAugmentationBatchIterator, self).transform(Xb, yb)

        augmentation_params = {
            'zoom_range': (1.0, 1.3),
            'rotation_range': (0, 360),
            'shear_range': (0, 20),
            'translation_range': (-5, 5),
        }

        IMAGE_WIDTH = PIXELS
        IMAGE_HEIGHT = PIXELS

        def fast_warp(img, tf, output_shape=(PIXELS,PIXELS), mode='constant'):
            """
            This wrapper function is about five times faster than skimage.transform.warp, for our use case.
            """
            #m = tf._matrix
            m = tf.params
            img_wf = np.empty((output_shape[0], output_shape[1]), dtype='float32')
            #for k in xrange(1):
            #    img_wf[..., k] = skimage.transform._warps_cy._warp_fast(img[..., k], m, output_shape=output_shape, mode=mode)
            img_wf = skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode)
            return img_wf

        def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True):
            # random shift [-10, 10] - shift no longer needs to be integer!
            shift_x = np.random.uniform(*translation_range)
            shift_y = np.random.uniform(*translation_range)
            translation = (shift_x, shift_y)

            # random rotation [0, 360]
            rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

            # random shear [0, 20]
            shear = np.random.uniform(*shear_range)

            # random zoom [0.9, 1.1]
            # zoom = np.random.uniform(*zoom_range)
            log_zoom_range = [np.log(z) for z in zoom_range]
            zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
            # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.


            translation = (0,0)
            rotation = 0.0
            shear = 0.0
            zoom = 1.0

            rotate =  np.random.randint(6)
            if rotate == 0:
                rotation = 0.0
            elif rotate == 1:
                rotation = 45.0
            elif rotate == 2:
                rotation = 90.0
            elif rotate == 3:
                rotation = 135.0
            elif rotate == 4:
                rotation = 180.0
            else:
                rotation = 270.0


            ## flip
            if do_flip and (np.random.randint(2) > 0): # flip half of the time
                shear += 180
                rotation += 180
                # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
                # So after that we rotate it another 180 degrees to get just the flip.            

            '''
            print "translation = ", translation
            print "rotation = ", rotation
            print "shear = ",shear
            print "zoom = ",zoom
            print ""
            '''

            return build_augmentation_transform(zoom, rotation, shear, translation)


        center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
        tform_center = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
            tform_augment = transform.AffineTransform(scale=(1/zoom, 1/zoom), 
                                                      rotation=np.deg2rad(rotation), 
                                                      shear=np.deg2rad(shear), 
                                                      translation=translation)
            tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
            return tform

        tform_augment = random_perturbation_transform(**augmentation_params)
        tform_identity = skimage.transform.AffineTransform()
        tform_ds = skimage.transform.AffineTransform()
        
        for i in range(Xb.shape[0]):
            new = fast_warp(Xb[i][0], tform_ds + tform_augment + tform_identity, output_shape=(PIXELS,PIXELS), mode='constant').astype('float32')
            Xb[i,:] = new
            #plot_sample(Xb[i])
            
        return Xb, yb
        
        
Maxout = layers.pool.FeaturePoolLayer
conv_net = NeuralNet(
    layers = [
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        #('maxout4', Maxout),
        ('hidden5', layers.DenseLayer),
        ('dropout5', layers.DropoutLayer),
        #('maxout5', Maxout),
        ('output', layers.DenseLayer),
    ],
    input_shape = (None, 1, PIXELS, PIXELS),
    conv1_num_filters = 32, conv1_filter_size = (3, 3), pool1_ds = (2, 2),
    dropout1_p = 0.1,
    conv2_num_filters = 64, conv2_filter_size = (3, 3), pool2_ds = (2, 2),
    dropout2_p = 0.2,
    conv3_num_filters = 128, conv3_filter_size = (3, 3), pool3_ds = (2, 2),
    dropout3_p = 0.3,
    
    hidden4_num_units = 512,
    dropout4_p = 0.5,
    #maxout4_ds = 2,
    
    hidden5_num_units = 512,
    dropout5_p = 0.5,
    #maxout5_ds = 2,
    
    output_num_units = 5, # 5 target values
    output_nonlinearity = lasagne.nonlinearities.softmax, # output layer uses sigmoid function
    
    # optimization method
    #update = rmsprop,
    update_momentum = theano.shared(float32(0.9)),
    update_learning_rate = theano.shared(float32(learning_rate)),
    
    batch_iterator_train = DataAugmentationBatchIterator(batch_size = 128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start = learning_rate, stop = 0.0001),
        #AdjustVariable('update_momentum', start = 0.9, stop = 0.999),
        #EarlyStopping(patience = 200),
        ],
    max_epochs = 5000,
    regression = False,
    verbose = 1,
    eval_size = 0.2
    )
    
    
def main():
    
    train_images, y = load2d(PATH, yPATH, test = False)
    
    # sanity check
    print train_images.shape
    print train_images[0].shape
    print y
    print np.amax(y)
    plot_sample(train_images[randint(0, train_images.shape[0])])
    plot_sample(train_images[randint(0, train_images.shape[0])])
    plot_sample(train_images[randint(0, train_images.shape[0])])
    plot_sample(train_images[randint(0, train_images.shape[0])])
    
    # binarize labels
    label_enc = LabelEncoder() #LabelBinarizer()
    y = label_enc.fit_transform(y)
    y = y.astype(np.int32)
    #print 'output shape'
    #print y.shape
    
    start = time.time()
    
    conv_net.fit(train_images, y)
    
    end = time.time()
    
    print 'Training time: ', (end - start) / 60
    plot_loss(conv_net)
    
    preds = conv_net.predict(train_images)
    preds = np.around(preds)
    
    print accuracy_score(y, preds)


if __name__ == '__main__':
    main()
