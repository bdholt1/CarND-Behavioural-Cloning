import numpy as np
from skimage import io, transform
import os
import glob
import h5py
import pandas
import math

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt

IMG_WIDTH = 160
IMG_HEIGHT = 80

def plot_histogram(y, label):
    fig = plt.figure()
    plt.hist(y, bins=100, label=label)
    plt.legend()
    plt.title("Histogram of steering angles")
    plt.show()


def rgb2yuv(rgb):
    conv = np.array([[0.299, 0.587, 0.114],
                     [-0.14713, -0.28886, 0.436],
                     [0.615, -0.51499, -0.10001]])
    return np.dot(rgb, conv.T)

def load_data(dir, csv, offset):
    """
    Load the images and associated steering angles.
    """
    imgs = []
    angles = []
    dataframe = pandas.read_csv(csv, header=None)
    dataset = dataframe.values
    center_imgs = dataset[:,0]
    left_imgs = dataset[:,1]
    right_imgs = dataset[:,2]
    steering_angles = dataset[:,3] # steering angle
    for center, left, right, angle in zip(center_imgs, left_imgs, right_imgs, steering_angles):
        # discard the path - data was captured on multiple machines
        # and the models run on multiple machines
        path, center_file = os.path.split(center)
        path, left_file = os.path.split(left)
        path, right_file = os.path.split(right)

        # discard datapoints that have a steering angle of 0
        # otherwise the system becomes overly weighted to going straight
        if math.isclose(angle, 0, abs_tol=0.001):
            continue

        #print("file: ", dir + "/" + file, ",  angle: ", angle)
        # center image
        imgs.append(rgb2yuv(transform.resize(io.imread(dir + "/" + center_file), (IMG_HEIGHT, IMG_WIDTH))))
        angles.append(angle)

        # left image
        imgs.append(rgb2yuv(transform.resize(io.imread(dir + "/" + left_file), (IMG_HEIGHT, IMG_WIDTH))))
        angles.append(angle - offset)

        # right image
        imgs.append(rgb2yuv(transform.resize(io.imread(dir + "/" + right_file), (IMG_HEIGHT, IMG_WIDTH))))
        angles.append(angle + offset)

    X = np.array(imgs, dtype='float32')
    Y = np.array(angles, dtype='float32')
    print("Loaded", X.shape[0], " files from ", dir)

    plot_histogram(Y, dir)

    return X, Y

# Augment data for training using this configuration
train_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        width_shift_range=0.1,
        height_shift_range=0.01,
        rescale=1./255,
        fill_mode='nearest')

# Training data is from the left track
X_train, Y_train = load_data("TRAIN", "train_driving_log.csv", 0.1)

train_generator = train_datagen.flow(X_train, Y_train, batch_size=128) #, save_to_dir="TRAIN_AUG/")

# For validation and test normlise the images
valid_test_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rescale=1./255)

# Validation data is from the left track (1 lap)
X_valid, Y_valid = load_data("VALID", "valid_driving_log.csv", 0.1)

validation_generator = valid_test_datagen.flow(X_valid, Y_valid, batch_size=128)#, save_to_dir="VALID_AUG/")


def cnn_model():
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    print("Using CNN model")
    model.summary()
    return model

model = cnn_model()

model.fit_generator(
        train_generator,
        samples_per_epoch=20000,
        nb_epoch=20,
        validation_data=validation_generator,
        nb_val_samples=2000)


# save model
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    outfile.write(json_string)

#save weights
model.save_weights("model.h5")
