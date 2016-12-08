import numpy as np
from skimage import io
import os
import glob
import h5py
import pandas

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt

IMG_WIDTH = 320
IMG_HEIGHT = 160

def load_data(dir, csv):
    """
    Load the images and associated steering angles.
    """
    imgs = []
    angles = []
    dataframe = pandas.read_csv(csv, header=None)
    dataset = dataframe.values
    img_paths = dataset[:,0] # use the centre image
    steering_angles = dataset[:,3] # steering angle
    for img_path, angle in zip(img_paths, steering_angles):
        # discard the path - data was captured on multiple machines
        # and the models run on multiple machines 
        path, file = os.path.split(img_path)
        #print("file: ", dir + "/" + file, ",  angle: ", angle)
        img = io.imread(dir + "/" + file)
        imgs.append(img)
        angles.append(angle)

    X = np.array(imgs, dtype='float32')
    Y = np.array(angles, dtype='float32')
    print("Loaded files from ", dir)
    
    return X, Y    

# Augment data for training using this configuration
train_datagen = ImageDataGenerator(
        rotation_range=5,
	    width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.01,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Training data is from the left track
X_train, Y_train = load_data("LEFT", "left_driving_log.csv")

train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)

# For validation (and test) the only thing we're going to do is rescale
validation_datagen = ImageDataGenerator(rescale=1./255)

# Validation data is from the right track
X_valid, Y_valid = load_data("RIGHT", "right_driving_log.csv")

validation_generator = validation_datagen.flow(X_valid, Y_valid, batch_size=32)


def baseline_model():
    # create model
    model = Sequential()
    model.add(Reshape((IMG_WIDTH*IMG_HEIGHT*3,), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print("Using MLP model")
    return model

def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    print("Using CNN model")
    return model

model = cnn_model()

model.fit_generator(
        train_generator,
        samples_per_epoch=2000, #2000
        nb_epoch=10, #50
        validation_data=validation_generator,
        nb_val_samples=800)  #800

model.summary()
# save model
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    outfile.write(json_string)

#save weights
model.save_weights("model.h5")
