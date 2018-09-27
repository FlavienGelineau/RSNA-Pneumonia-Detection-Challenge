import keras
import tensorflow as tf
from keras.layers import (BatchNormalization, Concatenate, Conv2D,
                          Conv2DTranspose, Dropout, LeakyReLU, MaxPool2D,
                          UpSampling2D)


def create_downsample(channels, inputs):
    x = BatchNormalization(momentum=0.9)(inputs)
    x = LeakyReLU(0)(x)
    x = Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = MaxPool2D(2)(x)
    return x


def create_resblock(channels, inputs):
    x = BatchNormalization(momentum=0.9)(inputs)
    x = LeakyReLU(0)(x)
    x = Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])


# local network use high-res moving window
def create_local_network(channels, input, depth, n_blocks):
    x = Conv2D(channels, 3, padding='same', use_bias=False)(input)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(256, 1, activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2DTranspose(filters=128,
                        kernel_size=(8, 8),
                        strides=(4, 4),
                        padding="same", activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(1, 1, activation=None)(x)
    return x


# subnetworks that handle low-resolution global image
def create_subnetwork(channels, input, depth, n_blocks):
    x = Conv2D(channels, 3, padding='same', use_bias=False)(input)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(256, 1, activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2DTranspose(filters=128,
                        kernel_size=(8, 8),
                        strides=(4, 4),
                        padding="same", activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(1, 1, activation=None)(x)
    return x


# Network that handles position in the global image
def create_global_network(channels, input1, input2, depth1, n_blocks1, depth2, n_blocks2):
    x = create_subnetwork(channels, input1, depth1, n_blocks1)
    y = create_subnetwork(channels, input2, depth2, n_blocks2)
    x = Concatenate([x, y])
    x = Conv2D(channels, 3, padding='same', use_bias=False)(input)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(channels, 3, padding='same', use_bias=False)(input)
    return x


def create_network(local_input_size, global_input_size, channels, n_blocks=2, depth=4):
    # inputs
    local_inputs = keras.Input(shape=(local_input_size, local_input_size, 1))
    global_map_inputs = keras.Input(
        shape=(global_input_size, global_input_size, 1))
    global_position_inputs = keras.Input(
        shape=(global_input_size, global_input_size, 1))

    local_net = create_local_network(channels, local_inputs, depth, n_blocks)
    global_net = create_global_network(channels, global_map_inputs, global_position_inputs, depth, n_blocks, depth,
                                       n_blocks)
    x = Concatenate([local_net, global_net])
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(256, 1, activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2DTranspose(filters=128,
                        kernel_size=(8, 8),
                        strides=(4, 4),
                        padding="same", activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(1, 1, activation='sigmoid')(x)
    outputs = UpSampling2D(2 ** (depth - 2))(x)
    model = keras.Model(
        inputs=[local_inputs, global_map_inputs, global_position_inputs], outputs=outputs)
    return model

# Focal loss for dynamically adjusted loss function
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) +
                                   tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score


# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * focal_loss(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)


# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(
        y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
