import tensorflow as tf
from tensorflow.data import Dataset, Iterator
from datagenerator4prediction_0814 import PreDataGenerator
from datetime import datetime
import os
import numpy as np
from alexnet import AlexNet
import re
import math

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# use the specified gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

l = 0
tf.flags.DEFINE_integer('pre_size', 4800, 'prediction size')  # todo:oom
tf.flags.DEFINE_integer('iter_epoch', l, 'pre_size data per iter_epoch')
FLAGS = tf.flags.FLAGS

# Path to the text files for the prediction set
pre_file = './pre_list_70500.csv'  # todo:
model_path = './Checkpoints_5c/model_epoch57.ckpt'  # '../Checkpoints_all/0705/model_epoch18.ckpt'

output_dir = './PreResult'
# Create output path if it doesn't exist
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
output_file = open(os.path.join(output_dir,
                                "prediction_cp30.txt"), 'a+')  # todo:

output_file.write('Sample, ' + 'Pred_1,' + '\n')

batch_size = 32  # todo:
num_classes = 2
depth_num = 6

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    pre_data = PreDataGenerator(pre_file,
                                mode='testing',  # 'predicting',
                                batch_size=batch_size,
                                num_classes=num_classes,
                                iterator_size=FLAGS.pre_size,
                                kth_init_op=FLAGS.iter_epoch,
                                classifier_version=2,
                                depth_num=depth_num)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(pre_data.data.output_types,
                                       pre_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
predicting_init_op = iterator.make_initializer(pre_data.data)

# #---------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, 227, 227, depth_num])
keep_prob = tf.constant(1., dtype=tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, 2, [], weight_decay=0., moving_average_decay=0., frozen_layer=[])

# Link variable to model output
score = model.fc8

softmax = tf.nn.softmax(score)

# create saver instance
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
pre_steps = int(np.floor(pre_data.data_size / batch_size))

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    saver.restore(sess, model_path)  # todo:

    print(pre_steps)
    print("{} Start predicting...".format(datetime.now()))

    # Initialize iterator with the predicting dataset
    sess.run(predicting_init_op)

    count = 0 + FLAGS.iter_epoch * FLAGS.pre_size
    for i in range(pre_steps):
        # (FLAGS.iter_epoch * FLAGS.pre_size, FLAGS.iter_epoch * FLAGS.pre_size + FLAGS.pre_size):
        #:
        # get next batch of data
        print("{} batches already fed : {:.0f} - {}".format(datetime.now(), i, count))

        # if not count > 1000:
        img_batch = sess.run(next_batch)

        # continue
        # And run the predicting op
        img_batch = tf.reshape(img_batch, (batch_size, 227, 227, depth_num))
        pred = sess.run(softmax, feed_dict={x: sess.run(img_batch)})
        for j in range(batch_size):
            output_file.write(str(count) + ' , ' + str(pred[j][1]) + '\n')
            # msg = str(count) + ' , ' + str(pred[j][0]) + ' , ' + str(pred[j][1]) + '\n'
            # print(msg)
            count = count + 1
            # sess.run(pred2, feed_dict={x2: sess.run(img_batch)})

        else:
            continue
            # count += 1
            print(str(count) + '  remainder')
            # continue
            for k, image in enumerate(imgs):
                height = 227
                width = 227
                depth = depth_num
                bytes = tf.decode_raw(image, out_type=tf.float32)
                img = tf.reshape(bytes, [1, height, width, depth])
                pred = sess.run(softmax, feed_dict={x: sess.run(img)})
                output_file.write(str(count) + ' , ' + str(pred[j][0]) + ' , ' + str(pred[j][1]) + '\n')
                count = count + 1

# tf.reset_default_graph()
output_file.close()
