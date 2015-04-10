import os
import numpy as np
import pandas as pd
import cPickle as pickle
from natsort import natsorted
from random import randint

from skimage import exposure
from matplotlib import pyplot
from skimage.io import imread
from skimage.io import imshow
from skimage.filters import sobel
from skimage import feature
from skimage.restoration import denoise_tv_chambolle

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

PATH = '/Volumes/Mildred/Kaggle/Retna/Data/Backup/train/'
LABELS = '/Volumes/Mildred/Kaggle/Retna/trainLabels.csv'

maxPixel = 196
imageSize = maxPixel * maxPixel
num_features = imageSize

def plot_sample(x):
    img = x.reshape(maxPixel, maxPixel) #/ 255.
    imshow(img)
    pyplot.show()

def load_images(path, labels):
    print 'reading file names ... '
    names = [d for d in os.listdir (path) if d.endswith('.jpeg')]
    names = natsorted(names)
    num_rows = len(names)
    print num_rows
    
    print 'making dataset ... '
    train_image = np.zeros((num_rows, num_features), dtype = float)
    levels = np.zeros((num_rows, 1), dtype = int)
    file_names = []
    i = 0
    for n in names:
        print n.split('.')[0], labels.level[labels.image == n.split('.')[0]].values
        
        image = imread(os.path.join(path, n))[:, :, 1]
        
        image = exposure.equalize_hist(image)
        #image = denoise_tv_chambolle(image, weight = 0.05, multichannel = False)

        train_image[i, 0:num_features] = np.reshape(image, (1, num_features))
        
        levels[i] = labels.level[labels.image == n.split('.')[0]].values
        i += 1
        
    return train_image, levels

labels = pd.read_csv(LABELS)
    
train, levels = load_images(PATH, labels)
train, levels = shuffle(train, levels)

print train[0]
print train.shape
print levels.shape

np.save('train_grey.npy', train)
np.save('train_y.npy', levels)

plot_sample(train[randint(0, train.shape[0])])
plot_sample(train[randint(0, train.shape[0])])
plot_sample(train[randint(0, train.shape[0])])
plot_sample(train[randint(0, train.shape[0])])
print np.amax(train[0])
print np.amin(train[0])

#print file_names

