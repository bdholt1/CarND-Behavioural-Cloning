Is the solution design documented?
The README thoroughly discusses the approach taken for deriving and designing a
 model architecture fit for solving the given problem.
 
This is my attempt at cloning my simulator driving skills using deep learning to mimic
behaviour to steer autonomously.

I chose to use Keras (https://keras.io) to develop the model because Keras offers some
very useful features that make the code quite concise, such as dataflow generators.

My first attempt (as most would do) was to train a 2 layer MLP just to get an indication
of how well the system was likely to do on the data.  On this very simple network I was
able to get validation errors of down to X.

The step up in the model is to use a CNN.  Here I chose to use 3 layers with activation,
maxpooling and dropout.  
kernel size, maxpool size, dropout rate?  
 


Is the model architecture documented?
The README provides sufficient details of the characteristics and qualities of 
the architecture, such as the type of model used, the number of layers, 
the size of each layer. Visualizations emphasizing particular qualities 
of the architecture are encouraged.

The final model uses a CNN with 3 convolution layers, each followed by a non-linear
activation layer, a maxpool layer and a dropout layer (to prevent overfitting). Each of
the convolutional layers progressively increases the depth from 32 to 64 to 128.

The final convolution output is flattened and fully connected with a 512 node hidden layer,
which is then condensed into a single output node.  

The model uses a mean squared error loss function (common to regression) and an
Adam optimiser.



Is the creation of the training dataset and training process documented?
The README describes how the model was trained and what the characteristics 
of the dataset are. Information such as how the dataset was generated and 
examples of images from the dataset should be included.

Training and validation data was captured by driving each of the left and right tracks
for 2 laps in both the "forwards" direction (the default direction the simulator sets
the vehicle in when starting) and "backwards".  Special effort was made to
drive offroad (while not recording) and then driving back onto the road (while recording).

This resulted in 3357 images from the left track and 2948 images from the right track.

The left track is used for training and the right track is help out for validation.
Nothing is used for testing.
