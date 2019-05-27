import tensorflow as tf
from tensorflow.data import Dataset, Iterator
from datagenerator4prediction import PreDataGenerator
from datetime import datetime

dir(tf.contrib)
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file("model_epoch10.ckpt", tensor_name='', all_tensors=True)
import os
import cv2
import numpy as np
from alexnet import AlexNet

# use the specified gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Path to the textfiles for the prediction set
sample_file = './tps_list.csv'  # todo:
count = len(open(sample_file, 'r').readlines())
samlen = 2000

# initialize input placeholder to specific batch_size, e.g. 1 if you want to classify image by image  output_file = open("TESTING123.txt",'w')
output_file = open("probabilities_tps.txt", 'w')  # todo:output list

output_file.write('Sample , ' + 'original label , ' + '0_prob , ' + '1_prob , ' '\n')

batch_size = 1  # todo:
num_classes = 2

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    pre_data = PreDataGenerator(sample_file,
                                mode='predicting',
                                batch_size=batch_size,
                                num_classes=num_classes,
                                iterator_size=samlen,
                                kth_init_op=1,
                                classifier_version=5
                                )

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(pre_data.data.output_types,
                                       pre_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
predicting_init_op = iterator.make_initializer(pre_data.data)

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 8])
keep_prob = tf.constant(1., dtype=tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, 2, [])

# Link variable to model output
score = model.fc8
softmax = tf.nn.softmax(score)

# create saver instance
saver = tf.train.Saver()
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
    saver.restore(sess, './checkpoints_tps/model_epoch123.ckpt')  # todo:

    print("{} Start predicting...".format(datetime.now()))

    # Initialize iterator with the predicting dataset
    sess.run(predicting_init_op)

    for i in range(pre_data.data_size):

        # get next batch of data
        img_batch = sess.run(next_batch)

        # And run the predicting op
        img_batch = tf.reshape(img_batch, (1, 227, 227, 8))
        pred = sess.run(softmax, feed_dict={x: sess.run(img_batch)})
        prob_0.append(pred[0][0])
        prob_1.append(pred[0][1])
        output_file.write(
            str(i) + ' , ' + str(pre_data.labels[i]) + ' , ' + str(prob_0[i]) + ' , ' + str(prob_1[i]) + '\n')
        # predicted_label = pred.argmax(axis=1)
        # predictions.append(predicted_label[0])
        # output_file.write(str(i) + ' , ' + str(predicted_label[0]) + '\n')

        if i & 0xFF == 0xFF:
            print("{} data already fed = {:.0f}".format(datetime.now(),
                                                        i))
tf.reset_default_graph()
output_file.close()
