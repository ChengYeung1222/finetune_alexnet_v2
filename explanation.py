import tensorflow as tf
from tensorflow.data import Dataset, Iterator
from datagenerator4prediction import PreDataGenerator
from datetime import datetime
import os
import numpy as np
from alexnet import AlexNet
import re
import math
import integrated_gradients_tf as ig

# use the specified gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

l = 0
tf.flags.DEFINE_integer('pre_size', 1600, 'prediction size')  # todo:oom
tf.flags.DEFINE_integer('iter_epoch', l, 'pre_size data per iter_epoch')
FLAGS = tf.flags.FLAGS

# Path to the textfiles for the prediction set
pre_file = './List/jj_unknow_pred_list.csv'  # todo:

# initialize input placeholder to specific batch_size, e.g. 1 if you want to classify image by image  output_file = open("TESTING123.txt",'w')
output_dir = './PreResult0705'
# Create output path if it doesn't exist
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

output_file = open(os.path.join(output_dir, "pre_jj_unknow_{}-{}.csv".format(FLAGS.iter_epoch * FLAGS.pre_size,
                                                                             FLAGS.iter_epoch * FLAGS.pre_size + FLAGS.pre_size - 1)),
                   'w')  # todo:ouput list

output_file.write('Sample, ' + 'Pred_0,' + 'Pred_1' + '\n')

batch_size = 1  # todo:
num_classes = 2
num_channels = 11

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    pre_data = PreDataGenerator(pre_file,
                                mode='predicting',
                                batch_size=batch_size,
                                num_classes=num_classes,
                                iterator_size=FLAGS.pre_size,
                                kth_init_op=FLAGS.iter_epoch,
                                classifier_version=4,
                                )

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(pre_data.data.output_types,
                                       pre_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
predicting_init_op = iterator.make_initializer(pre_data.data)

x = tf.placeholder(tf.float32, [None, 227, 227, num_channels])
inter, stepsize, ref = ig.linear_inpterpolation(x, num_steps=50)
keep_prob = tf.constant(1., dtype=tf.float32)

# Initialize model
# model = AlexNet(x, keep_prob, 2, [])
model_ig = AlexNet(inter, keep_prob, 2, [])

# Link variable to model output
# score = model.fc8
score_ig = model_ig.fc8

# softmax = tf.nn.softmax(score)
softmax_ig = tf.nn.softmax(score_ig)

# Calculate integrated gradients
explanations=[]
for i in range(num_classes):
    explanations.append(ig.build_ig(inter, stepsize, softmax_ig[:, i], num_steps=50))

# create saver instance
saver = tf.train.Saver()
# predictions = []
prob_0 = []
prob_1 = []
# Get the number of training/validation steps per epoch
pre_steps = int(np.floor(pre_data.data_size / batch_size))

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    saver.restore(sess, '../Checkpoints_all/model_epoch12.ckpt')  # todo:

    print("{} Start predicting...".format(datetime.now()))

    # Initialize iterator with the predicting dataset
    sess.run(predicting_init_op)

    count = 0
    for i in range(FLAGS.iter_epoch * FLAGS.pre_size, FLAGS.iter_epoch * FLAGS.pre_size + FLAGS.pre_size):
        # get next batch of data
        img_batch = sess.run(next_batch)

        # continue

        # And run the predicting op
        img_batch = tf.reshape(img_batch, (batch_size, 227, 227, num_channels))
        pred_ig = sess.run(explanations, feed_dict={x: sess.run(img_batch)})

        if i & 0xFF == 0xFF:
            print("{} batches already fed = {:.0f}".format(datetime.now(), i))


output_file.close()
