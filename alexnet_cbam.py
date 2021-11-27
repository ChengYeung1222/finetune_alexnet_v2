"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np

class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes, skip_layer, weight_decay, moving_average_decay, frozen_layer,
                 weights_path='DEFAULT'):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.FROZEN_LAYER = frozen_layer
        self.WEIGHT_DECAY = weight_decay
        self.MOVING_AVERAGE_DECAY = moving_average_decay

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""

        #cbam新
        # cbam = cbam_module(self.X, reduction_ratio=0.5, name="")
        # ca=cbam[0]

        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1', weight_decay=self.WEIGHT_DECAY)
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2', weight_decay=self.WEIGHT_DECAY)
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3', weight_decay=self.WEIGHT_DECAY)

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4', weight_decay=self.WEIGHT_DECAY)

        # CBAM1  #新wyf
        # cbam1 = cbam_module(conv4, reduction_ratio=0.5, name="1")

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5', weight_decay=self.WEIGHT_DECAY)

        # CBAM2  #新wyf
        # cbam2 = cbam_module(conv5, reduction_ratio=0.5, name="2")

        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6', weight_decay=self.WEIGHT_DECAY)
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7', weight_decay=self.WEIGHT_DECAY)
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8', weight_decay=self.WEIGHT_DECAY)

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes',allow_pickle=True).item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        # if op_name not in self.FROZEN_LAYER:
                        #     trainable = True
                        # else:
                        #     trainable = False

                        # Biases
                        trainable = [True if op_name not in self.FROZEN_LAYER else False][0]
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases',
                                                  trainable=False)  # todo: trainable
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights',
                                                  trainable=[True if op_name not in self.FROZEN_LAYER else False][0])
                            session.run(var.assign(data))


def get_weight(shape, lambda1):
    var = tf.get_variable(name='weights', shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer(),
                          dtype=tf.float32, trainable=True)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, weight_decay,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    :rtype: object
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        # weights = tf.get_variable('weights', shape=[filter_height,
        #                                             filter_width,
        #                                             input_channels / groups,
        #                                             num_filters])
        weights = get_weight(shape=[filter_height, filter_width, int(input_channels / groups), num_filters],
                             lambda1=weight_decay)  # todo
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)


    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    # bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    # relu = tf.nn.relu(bias, name=scope.name)

    relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

    return relu


def fc(x, num_in, num_out, name, weight_decay, relu=True, ):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = get_weight(shape=[num_in, num_out], lambda1=weight_decay)  # todo
        # weights = tf.get_variable('weights', shape=[num_in, num_out],
        #                           trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)


def cbam_module(inputs, reduction_ratio=0.8, name=""):
    with tf.variable_scope("cbam_" + name, reuse=tf.AUTO_REUSE):
        #个数/通道数
        batch_size, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]
        #先行求后列最大值,均值
        maxpool_channel = tf.reduce_max(tf.reduce_max(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
        avgpool_channel = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
        #一维展开
        maxpool_channel = tf.layers.Flatten()(maxpool_channel)
        avgpool_channel = tf.layers.Flatten()(avgpool_channel)
        #全连接
        mlp_1_max = tf.layers.dense(inputs=maxpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
                                    reuse=None, activation=tf.nn.relu)
        mlp_2_max = tf.layers.dense(inputs=mlp_1_max, units=hidden_num, name="mlp_2", reuse=None)
        mlp_2_max = tf.reshape(mlp_2_max, [batch_size, 1, 1, hidden_num])

        mlp_1_avg = tf.layers.dense(inputs=avgpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
                                    reuse=True, activation=tf.nn.relu)
        mlp_2_avg = tf.layers.dense(inputs=mlp_1_avg, units=hidden_num, name="mlp_2", reuse=True)
        mlp_2_avg = tf.reshape(mlp_2_avg, [batch_size, 1, 1, hidden_num])
        #求和sigmoid
        channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
        channel_refined_feature = inputs * channel_attention
        #空间注意力
        #通道维度求最大值,均值
        maxpool_spatial = tf.reduce_max(inputs, axis=3, keepdims=True)
        avgpool_spatial = tf.reduce_mean(inputs, axis=3, keepdims=True)
        #向量拼接
        max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
        #卷积,不改变图像长宽
        conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
                                      activation=None)
        #通过激活函数
        spatial_attention = tf.nn.sigmoid(conv_layer)

        refined_feature = channel_refined_feature * spatial_attention

    return channel_attention,spatial_attention,refined_feature
