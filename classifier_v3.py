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

import re

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# use the specified gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Path to the textfiles for the prediction set
pre_file = './data_6c_pre_list_1 .csv'  # todo:
count=len(open(pre_file,'r').readlines())
iterator_size=1000
num_init_ops=int(np.floor(count / iterator_size))

# initialize input placeholder to specific batch_size, e.g. 1 if you want to classify image by image  output_file = open("TESTING123.txt",'w')
output_file = open("prediction_g1_cp46.txt", 'w')  # todo:ouput list

output_file.write('Sample , ' + 'Prediction' + '\n')

batch_size = 1  # todo:
num_classes = 2

pre_data = []
predicting_init_op = []
pre_iterator=[]

# Place data loading and preprocessing on the cpu
for k in range(num_init_ops):
    with tf.device('/cpu:0'):


        pre_data.append(PreDataGenerator(pre_file,
                                    mode='predicting',
                                    batch_size=batch_size,
                                    num_classes=num_classes,
                                    shuffle=False,
                                    iterator_size=iterator_size,
                                    kth_init_op=k).data)

        pre_iterator.append(PreDataGenerator(pre_file,
                                    mode='predicting',
                                    batch_size=batch_size,
                                    num_classes=num_classes,
                                    shuffle=False,
                                    iterator_size=iterator_size,
                                    kth_init_op=k).iterator)

        next_batch=pre_iterator[k].get_next()

    #     # create an reinitializable iterator given the dataset structure
    #     iterator = Iterator.from_structure(pre_data[k].data.output_types,
    #                                        pre_data[k].data.output_shapes)
    #     next_batch = iterator.get_next()
    #
    # # Ops for initializing the two different iterators
    # predicting_init_op.append(iterator.make_initializer(pre_data[k].data))

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 6])
keep_prob = tf.constant(1., dtype=tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, 2, [])

# Link variable to model output
score = model.fc8
softmax = tf.nn.softmax(score)

# create saver instance
saver = tf.train.Saver()
predictions = []

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    saver.restore(sess, './checkpoints_grade1/model_epoch46.ckpt')  # todo:

    print("{} Start predicting...".format(datetime.now()))

    for j in range(num_init_ops+1):#todo:

        print('{} Initializing {} iterator'.format(datetime.now(),j))

        # Initialize iterator with the predicting dataset
        # sess.run(predicting_init_op[j])
        sess.run(pre_iterator[j].initializer)

        for i in range(iterator_size):

            while True:
                try:
                    # get next batch of data
                    img_batch = sess.run(next_batch)  # todo:?
                    # And run the predicting op
                    img_batch = tf.reshape(img_batch, (1, 227, 227, 6))
                    pred = sess.run(softmax, feed_dict={x: sess.run(img_batch)})
                    predicted_label = pred.argmax(axis=1)
                    predictions.append(predicted_label[0])
                    output_file.write(str(i) + ' , ' + str(predicted_label[0]) + '\n')
                except tf.errors.OutOfRangeError:
                    sess.run(pre_iterator[j].initializer)
                    # sess.run(predicting_init_op[j])
                    break



            if i & 0xFF == 0xFF:
                print("{} data already fed = {:.0f}".format(datetime.now(),
                                                        i))
tf.reset_default_graph()
output_file.close()
