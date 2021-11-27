import tensorflow as tf
from tensorflow.data import Dataset, Iterator
from datagenerator4prediction import PreDataGenerator
from datetime import datetime
import os
import csv  # new0926
import numpy as np
from alexnet_cbam import AlexNet
import re
import math
import integrated_gradients_tf as ig
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['KMP_WARNINGS'] = '0'

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# use the specified gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pre_size = 25
batch_size = 25
l = 0
num_classes = 2
num_channels = 20  # 23

tf.flags.DEFINE_integer('pre_size', pre_size, 'prediction size')  # todo:oom
tf.flags.DEFINE_integer('iter_epoch', l, 'pre_size data per iter_epoch')
FLAGS = tf.flags.FLAGS

# Path to the textfiles for the prediction set
if 1:  # todo dygz
    # pre_file = './List/dygz_train_20c_listall.csv'  # todo:
    pre_file = './List/dygz_val_20c_list.csv'
    model_path='./Model/DYGZ-20c-0.0001-200-MODEL-1111_fake!/checkpoints/model_epoch50.ckpt'
    # model_path ='./Model/10_percent/10_percent!/DYGZ-20c-0.0001-200-MODEL-1110_without_pretraining/checkpoints/model_epoch50.ckpt'
    # model_path = './Model/half/DYGZ-20c-0.0001-200-MODEL-1108_with_pretraining/checkpoints/model_epoch50.ckpt'
    # model_path = './Model/10_percent/DYGZ-20c-0.0001-200-MODEL-1110_with_pretraining/checkpoints/model_epoch50.ckpt'
    # model_path = './Model/DYGZ-20c-0.0001-200-MODEL-0423/checkpoints/model_epoch50.ckpt'
    # model_path = './Model/DYGZ-20c-0.0001-200-MODEL-1102/checkpoints/model_epoch50.ckpt'    # model_path = './Model/oldsave/0911Checkpoints_jj11c/model_epoch28.ckpt'  # todo:
    output_dir = './PreResoult'
    # output_file = open(os.path.join(output_dir, "preResoult_20c_dygz_26940_10_percent_wopretrain.csv"), 'a+')  # todo:
    # output_file = open(os.path.join(output_dir, "preResoult_20c_dygz_26940_10_wopretrain_val.csv"), 'a+')
    output_file = open(os.path.join(output_dir, "preResoult_20c_dygz_26940_fake_withpretraining_val.csv"), 'a+')
    # output_file = open(os.path.join(output_dir, "preResoult_20c_dygz_26940_all_wopretrain_val.csv"), 'a+')
else:  # todo jiaojia
    pre_file = './LISTNEW/PRE/jj_knew_list78351.csv'  # todo:
    model_path = './Model/oldsave/0911Checkpoints_jj11c/model_epoch28.ckpt'  # './Model/jj-qjCheckpoints0926/model_epoch99.ckpt'  # todo:
    output_dir = './PreResoult'
    output_file = open(os.path.join(output_dir, "prediction_10000.csv"), 'a+')  # todo:

# Create output path if it doesn't exist
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# if FLAGS.iter_epoch == 0:
#     output_file.write('count'+','+'pre'+'\n')

# 预测余数问题 new0926
count = 0 + FLAGS.iter_epoch * FLAGS.pre_size
num_steps = count // batch_size
file = open(pre_file)
csv = csv.reader(file)

len = len(list(file))
rem = len % FLAGS.pre_size
zheng = len - rem
re_size = FLAGS.pre_size

if count >= zheng:  # todo:
    iterator_size = rem
    batch_size = 1
file.close()

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    pre_data = PreDataGenerator(pre_file,
                                mode='test',  # 'predicting',
                                batch_size=batch_size,
                                re_size=re_size,
                                num_classes=num_classes,
                                iterator_size=FLAGS.pre_size,
                                kth_init_op=FLAGS.iter_epoch,
                                classifier_version=2,
                                depth_num=num_channels)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(pre_data.data.output_types,
                                       pre_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
predicting_init_op = iterator.make_initializer(pre_data.data)

# #---------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, 227, 227, num_channels])
keep_prob = tf.constant(1., dtype=tf.float32)

# Initialize model
weight_decay = 1e-3
moving_average_decay = 0.99
train_layers = ['conv1', 'fc6', 'fc7', 'fc8']  # TODO: 'fc8'
all_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
frozen_layers = ['conv2', 'conv3']
model = AlexNet(x, keep_prob, num_classes, train_layers, weight_decay, moving_average_decay,
                frozen_layer=frozen_layers)  # NEW
# model = AlexNet(x, keep_prob, 2, [])

# Link variable to model output
score = model.fc8

softmax = tf.nn.softmax(score)
# ---------------------------------------------------------------------------------------------

# create saver instance
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True

# Get the number of training/validation steps per epoch
pre_steps = int(np.floor(pre_data.data_size / batch_size))

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    saver.restore(sess, model_path)  # todo:

    print("{} Iter {} Start predicting...".format(datetime.now(), FLAGS.iter_epoch + 1))

    # Initialize iterator with the predicting dataset
    sess.run(predicting_init_op)
    labelnum = 0
    for i in range(pre_steps):
        # get next batch of data
        img_batch = sess.run(next_batch)
        img_batch = tf.reshape(img_batch, (batch_size, 227, 227, num_channels))

        pred = sess.run(softmax, feed_dict={x: sess.run(img_batch)})

        print("{} batches already fed : {:.0f} - {}".format(datetime.now(),
                                                            num_steps + 1, count + batch_size))

        for j in range(batch_size):
            output_file.write(str(pre_data.labels[j].eval()) + ' , ' + str(pred[j][1]) + '\n')
            # output_file.write(str(count) + ' , ' + str(pred[j][1]) + ' , ' + str(pre_data.labels[labelnum].eval()) + '\n')
            count += 1
            labelnum += 1
        num_steps += 1

tf.reset_default_graph()
output_file.close()
