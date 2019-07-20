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

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


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

# #---------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, 227, 227, num_channels])
inter, stepsize, ref = ig.linear_inpterpolation(x, num_steps=50)
keep_prob = tf.constant(1., dtype=tf.float32)

# Initialize model
# model = AlexNet(x, keep_prob, 2, [])
model_ig = AlexNet(inter, keep_prob, 2, [])

# Remainder
current_dir = os.getcwd()
image_dir = os.path.join(current_dir, '../jj_100/')  # todo:'../../../../mnt/usbhd1/data/'
# img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.txt')]
imgs = []
groundtruth = []
# importedgraph = tf.train.import_meta_graph("model_epoch10.ckpt.meta")
temp_files = [image_dir + "/" + f for f in os.listdir(image_dir) if f.endswith('.bin')]
len_temp_files = len(temp_files)
num_iter = len_temp_files // batch_size
temp_files.sort(key=numericalSort)
remd = len_temp_files % batch_size
temp_files = temp_files[-remd:]
# print(temp_files)
for f in temp_files:
    if not f.endswith('.bin'):
        continue
    else:
        imgs.append(tf.read_file(f))

# Link variable to model output
# score = model.fc8
score_ig = model_ig.fc8

# softmax = tf.nn.softmax(score)
softmax_ig = tf.nn.softmax(score_ig)

# Calculate integrated gradients
explanations=[]
for i in range(num_classes):
    explanations.append(ig.build_ig(inter, stepsize, softmax_ig[:, i], num_steps=50))
# ---------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# integrated gradient
# x2 = tf.placeholder(tf.float32, [None, 227, 227, 11])
# _x2 = tf.contrib.slim.flatten(x2)
# inter, stepsize, ref = ig.linear_inpterpolation(_x2, num_steps=50)
#
# keep_prob2 = tf.constant(1., dtype=tf.float32)
# # integrated gradient
# model2 = AlexNet(x2, keep_prob2, 2, [])
#
# # Link variable to model output
# score2 = model2.fc8
# pred2 = tf.nn.softmax(score2)
#
# ig.build_ig(inter, stepsize, pred2[:, 0], num_steps=50)
# --------------------------------------------------------------------------------------

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
        print("{} batches already fed = {:.0f}".format(datetime.now(), i))
        img_batch = sess.run(next_batch)

        # continue

        # And run the predicting op
        img_batch = tf.reshape(img_batch, (batch_size, 227, 227, num_channels))
        pred_ig = sess.run(explanations, feed_dict={x: sess.run(img_batch)})
        continue

        # pred = sess.run(softmax, feed_dict={x: sess.run(img_batch)})
        # predicted_label = pred.argmax(axis=1)
        # predictions.append(predicted_label[0])
        # output_file.write(str(i) + ' , ' + str(predicted_label[0]) + '\n')

        # prob_0.append(pred[0][0])
        # prob_1.append(pred[0][1])
        # output_file.write(str(i) + ' , '  + ' , ' + str(prob_1[i]) + ' , '+  str(prob_0[i]) +'\n')
        # if count <batch_size*num_iter:
        if count < 100000:
            # count += 1
            print(count)
            # continue
            img_batch = sess.run(next_batch)

            # continue

            # And run the predicting op
            img_batch = tf.reshape(img_batch, (batch_size, 227, 227, num_channels))
            pred = sess.run(softmax, feed_dict={x: sess.run(img_batch)})
            for j in range(batch_size):
                output_file.write(str(i) + ' , ' + str(pred[j][0]) + ' , ' + str(pred[j][1]) + '\n')
                # msg = str(count) + ' , ' + str(pred[j][0]) + ' , ' + str(pred[j][1]) + '\n'
                # print(msg)
                count = count + 1
            # sess.run(pred2, feed_dict={x2: sess.run(img_batch)})

        elif count == 10000000000:
            # count += 1
            print(str(count) + '  remainder')
            # continue
            for k, image in enumerate(imgs):
                height = 227
                width = 227
                depth = num_channels
                bytes = tf.decode_raw(image, out_type=tf.float32)
                img = tf.reshape(bytes, [1, height, width, depth])
                pred = sess.run(softmax, feed_dict={x: sess.run(img)})
                output_file.write(str(count) + ' , ' + str(pred[j][0]) + ' , ' + str(pred[j][1]) + '\n')
                count = count + 1
                # img_decoded = tf.image.decode_jpeg(image, channels=3)
                # img_resized = tf.image.resize_images(img_decoded, [227, 227])
                # img_centered = tf.subtract(img_resized,imagenet_mean)
                # img_centered = tf.subtract(img_centered,trainingset_mean)
                # img_bgr = img_centered[:, :, ::-1]
                # # Reshape as needed to feed into model

        else:
            pass

        # if i & 0xFF == 0xFF:

# tf.reset_default_graph()
output_file.close()
