# Import the necessary modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# Define the constants and hyperparameters
IMAGE_SIZE = 64 # the size of the input and output images
LATENT_DIM = 512 # the dimension of the latent space
STYLE_DIM = 512 # the dimension of the style vector
NUM_LAYERS = int(np.log2(IMAGE_SIZE)) - 1 # the number of layers in the generator and the discriminator
BATCH_SIZE = 32 # the batch size for training
LEARNING_RATE = 0.0001 # the learning rate for training
BETA_1 = 0.0 # the beta_1 parameter for the Adam optimizer
BETA_2 = 0.99 # the beta_2 parameter for the Adam optimizer
EPSILON = 1e-8 # the epsilon parameter for the Adam optimizer
W_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02) # the weight initializer for the model parameters
GAMMA_INIT = keras.initializers.RandomNormal(mean=1.0, stddev=0.02) # the gamma initializer for the adaptive instance normalization
MIXING_PROB = 0.9 # the probability of mixing styles for the generator

# Define the mapping network
def MappingNetwork():
  # The mapping network maps a latent vector to a style vector
  inputs = layers.Input(shape=(LATENT_DIM,)) # the input layer
  x = inputs
  for i in range(8): # the mapping network has 8 fully connected layers
    x = layers.Dense(STYLE_DIM, kernel_initializer=W_INIT)(x) # a dense layer
    x = layers.LeakyReLU(alpha=0.2)(x) # a leaky ReLU activation layer
  outputs = x
  return keras.Model(inputs, outputs, name="mapping_network") # return the mapping network model

# Define the adaptive instance normalization layer
def AdaIN(x, gamma, beta):
  # The adaptive instance normalization layer normalizes the input x and scales and shifts it with gamma and beta
  mean, variance = tf.nn.moments(x, axes=[1], keepdims=True) # compute the mean and variance of x along the channel axis
  x = (x - mean) / tf.sqrt(variance + EPSILON) # normalize x
  x = x * gamma + beta # scale and shift x
  return x # return the output

# Define the modulation layer
def ModulationLayer(units, name):
  # The modulation layer modulates the input style with the learned affine transformation
  return layers.Dense(units, kernel_initializer=W_INIT, name=name) # return a dense layer

# Define the noise layer
def NoiseLayer(inputs):
  # The noise layer adds random noise to the inputs
  noise = tf.random.normal(tf.shape(inputs)) # generate random noise with the same shape as the inputs
  weight = tf.Variable(tf.zeros([1, 1, inputs.shape[-1]]), name="weight") # create a weight variable for scaling the noise
  return inputs + noise * weight # return the output

# Define the convolution layer
def ConvLayer(filters, kernel_size, padding="same", upsample=False, downsample=False, apply_bias=True, name="conv"):
  # The convolution layer performs a convolution operation on the inputs, with optional upsampling or downsampling
  def layer(inputs):
    if upsample: # if upsampling is True
      inputs = layers.UpSampling2D()(inputs) # perform nearest neighbor upsampling on the inputs
    x = inputs
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=W_INIT, use_bias=apply_bias, name=name)(x) # perform a 2D convolution on x
    if downsample: # if downsampling is True
      x = layers.AveragePooling2D()(x) # perform average pooling on x
    return x # return the output
  return layer # return the convolution layer

# Define the residual block
def ResBlock(filters, name):
  # The residual block performs two convolution operations and adds the inputs to the outputs
  def layer(inputs):
    x = inputs
    x = layers.LeakyReLU(alpha=0.2)(x) # a leaky ReLU activation layer
    x = ConvLayer(filters, 3, name=name + "_conv1")(x) # a convolution layer
    x = layers.LeakyReLU(alpha=0.2)(x) # a leaky ReLU activation layer
    x = ConvLayer(filters, 3, name=name + "_conv2")(x) # a convolution layer
    return layers.Add()([inputs, x]) # return the sum of the inputs and x
  return layer # return the residual block

# Define the synthesis block
def SynthesisBlock(filters, name):
  # The synthesis block performs adaptive instance normalization, noise addition, and convolution on the inputs
  def layer(inputs, style1, style2, noise1, noise2):
    x = inputs
    x = AdaIN(x, style1[:, :filters], style1[:, filters:]) # perform adaptive instance normalization on x with style1
    x = NoiseLayer(x) # add noise to x
    x = layers.LeakyReLU(alpha=0.2)(x) # a leaky ReLU activation layer
    x = ConvLayer(filters, 3, upsample=True, name=name + "_conv1")(x) # a convolution layer with upsampling
    x = AdaIN(x, style2[:, :filters], style2[:, filters:]) # perform adaptive instance normalization on x with style2
    x = NoiseLayer(x) # add noise to x
    x = layers.LeakyReLU(alpha=0.2)(x) # a leaky ReLU activation layer
    x = ConvLayer(filters, 3, name=name + "_conv2")(x) # a convolution layer
    return x # return the output
  return layer # return the synthesis block

# Define the generator network
def GeneratorNetwork():
  # The generator network generates an image from a latent vector
  style_inputs = [] # a list to store the style inputs
  noise_inputs = [] # a list to store the noise inputs
  for i in range(NUM_LAYERS): # for each layer in the generator
    style_inputs.append(layers.Input(shape=(STYLE_DIM,))) # create a style input layer and append it to the list
    noise_inputs.append(layers.Input(shape=(IMAGE_SIZE // (2 ** i), IMAGE_SIZE // (2 ** i), 1))) # create a noise input layer and append it to the list
  x = layers.Input(shape=(1, 1, LATENT_DIM)) # the input layer
  x = layers.Dense(4 * 4 * (2 ** (NUM_LAYERS - 1)) * LATENT_DIM, kernel_initializer=W_INIT)(x) # a dense layer
  x = layers.Reshape((4, 4, (2 ** (NUM_LAYERS - 1)) * LATENT_DIM))(x) # a reshape layer
  x = ResBlock((2 ** (NUM_LAYERS - 1)) * LATENT_DIM, name="resblock")(x) # a residual block
  for i in range(1, NUM_LAYERS): # for each layer in the generator, except the first one
    x = SynthesisBlock(2 ** (NUM_LAYERS - i - 1) * LATENT_DIM, name="synthesisblock" + str(i))(x, style_inputs[i - 1], style_inputs[i], noise_inputs[i - 1], noise_inputs[i]) # a synthesis block
  x = layers.LeakyReLU(alpha=0.2)(x) # a leaky ReLU activation layer
  x = ConvLayer(3, 1, apply_bias=False, name="conv")(x) # a convolution layer
  x = layers.Activation("tanh")(x) # a tanh activation layer
  outputs = x
  return keras.Model([style_inputs, noise_inputs], outputs, name="generator_network") # return the generator network model

# Define the discriminator network
def DiscriminatorNetwork():
  # The discriminator network classifies an image as real or fake
  inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)) # the input layer
  x = inputs
  for i in range(NUM_LAYERS): # for each layer in the discriminator
    x = ConvLayer(2 ** (i + 1) * LATENT_DIM, 3, downsample=True, name="conv" + str(i))(x) # a convolution layer with downsampling
    x = layers.LeakyReLU(alpha=0.2)(x) # a leaky ReLU activation layer
  x = layers.Flatten()(x) # a flatten layer
  x = layers.Dense(1, kernel_initializer=W_INIT)(x) # a dense layer
  outputs = x
  return keras.Model(inputs, outputs, name="discriminator_network") # return the discriminator network model

# Define the loss function
def LossFunction():
  # The loss function computes the Wasserstein loss with gradient penalty for the generator and the discriminator
  def gradient_penalty(real_images, fake_images, discriminator):
    # The gradient penalty regularizes the discriminator to have a 1-Lipschitz constraint
    alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1]) # generate random coefficients for interpolation
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images # interpolate between real and fake images
    with tf.GradientTape() as tape: # create a gradient tape to record the operations
      tape.watch(interpolated_images) # watch the interpolated images
      interpolated_scores = discriminator(interpolated_images) # compute the discriminator scores for the interpolated images
    gradients = tape.gradient(interpolated_scores, interpolated_images) # compute the gradients of the scores with respect to the images
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])) # compute the norm of the gradients
    penalty = tf.reduce_mean((norm - 1.0) ** 2) # compute the penalty as the squared difference from 1
    return penalty # return the penalty
  def discriminator_loss(real_scores, fake_scores, penalty):
    # The discriminator loss is the difference between the scores for real and fake images, plus the gradient penalty
    return tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores) + 10 * penalty # return the discriminator loss
  def generator_loss(fake_scores):
    # The generator loss is the negative of the scores for fake images
    return -tf.reduce_mean(fake_scores) # return the generator loss
  return discriminator_loss, generator_loss # return the loss functions

# Define the optimizer
def Optimizer():
  # The optimizer updates the model parameters using the Adam algorithm
  return keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON) # return the optimizer
