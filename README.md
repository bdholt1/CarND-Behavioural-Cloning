# Behavioural Cloning

## Solution approach

This is my attempt at cloning my simulator driving skills using deep learning to mimic
behaviour to steer autonomously.

I chose to use Keras (https://keras.io) to develop the model because Keras offers some
very useful features that make the code quite concise, such as dataflow generators.

My model follows the architecture of the NVIDIA paper very closely in terms of structure,
with 5 convolutional layers and 4 fully connected layers.  NVIDIA uses images of 66x200
whereas I use 80x160 (resizing the originals from 160x320).  This means that I end up with slightly
more parameters than they did, but the architecture is the same.
Additionally I added 2 dropout layers in the fully-connected section to reduce overfitting.

The theory is that the convolutional layers learn features and the fully connected layers learn
to make a decision (output a value) based on those features.

My biggest source of trouble was attempting to normalise the data.  I originally tried rescaling
int Keras ImageDataGenerator and subsequently using a Keras Lambda to rescale, but after many
trials where it appeared that the network wasn't learning and this was very frustrating. It was
only after much debugging that I realised that scikit-image already scales the image to 0 - 1.

I experimented with alternative normalisation approaches, trying grey-only, yuv (note the rgb2yuv function)
and rgb.  The lighting is fairly consistent so there seems no real need for histogram normalisation.

I found that the steering angles from the data I collected (with a keyboard) were very extreme
and needed to be smoothed, and so another tune-able parameter is the number of samples to use when
computing the moving average. See driving_angles.png and driving_angles_smoothed.png for examples.

After computing the smoothed steering angles I filter out examples with a steering angle that is
very close to 0 because they dominate the training data otherwise. There are plots of the steering data
before and after smoothing and also a histogram of the steering angles that are used for training,
showing a Gaussian distribution - which is what you would expect. See driving_angles_histogram.png for
the plot.

I also found that steering angles close to 1 were very extreme and threw off the training,
so I eliminate them from the training set.

## Model architecture

The final model uses a CNN with 3 convolution layers, each followed by a non-linear
activation layer and a maxpool layer. Each of
the convolutional layers progressively increases the depth from 24 to 36 to 48.
A final pair of convolution layers at depth 64 with relu is added before flattening

The final convolution output is flattened and fully connected with a 1024 node hidden layer,
a 100 node layer, a 50 node layer, a 10 node layer and then a final output node.

The model uses a mean squared error loss function (common to regression) and an
Adam optimiser.

One of the really nice things about Keras is that the code derived from it is effectively 'self-documenting'.

This is the model: 3 convolutional layers with non-linear activation and maxpooling

    model.add(Convolution2D(24, 5, 5, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

followed by 2 convolutional layers with just non-linear activation

    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Flatten())

add dropout to combat overfitting

    model.add(Dropout(0.1))

fully connect to the final convolutional layer

    model.add(Dense(1024, activation='relu'))

another dropout to combat overfitting

    model.add(Dropout(0.1))

3 more fully connected layers leading to the output

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, init='normal'))

# Dataset creation

Training and validation data was captured by driving each of the left and right tracks
for 2 laps in both the "forwards" direction (the default direction the simulator sets
the vehicle in when starting) and "backwards".  Special effort was made to
drive offroad (while not recording) and then driving back onto the road (while recording).

This resulted in 3357 images from the left track and 2948 images from the right track.

It became clear that this would not be enough data so I drove the left track 5 more times,
and then drove parts of the tracks that the model was getting wrong so that the model
could work out what to do instead.

For validation I drove the left track again and held out that entire lap as validation data.

The data is augmented using an ImageDataGenerator with small width shifts and height shifts.
