"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os
from typing import Optional, Any

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.data import Iterator

"""
Configuration Part.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Path to the textfiles for the trainings and validation set
train_file = './Cut_off_grade_2_train_list.csv'  # todo:
val_file = './Cut_off_grade_2_val_list.csv'  # todo:

# Learning params
learning_rate = 1e-5  # TODO: decrease it
num_epochs = 500  # TODO :2:20
batch_size = 256  # TODO: 128

# Network params
dropout_rate = 0.5
num_classes = 2  # todo:
train_layers = ['conv1', 'fc8']  # TODO: 'fc8'
all_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./tensorboard_grade2_debugging"  # TODO
checkpoint_path = "./checkpoints_grade2_debugging"  # TODO

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()
#
# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 6])  # todo: input_size
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8  # todo:
# coe = tf.constant([1.0, 5.0])
# score_coe = score * coe

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in all_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                     labels=tf.clip_by_value(y, 1e-4,
                                                                                             tf.reduce_max(y))))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.AdamOptimizer(learning_rate)  # TODO: adam
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# with tf.name_scope("auc"):
#     softmax=tf.nn.softmax(score)
#     prediction_list = tf.placeholder(tf.float32, [batch_size, num_classes], name='prediction_list')
#     auc=tf.metrics.auc(labels=y,predictions=prediction_list)

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# tf.summary.scalar('auc', auc)
# tf.summary.scalar('auc',auc)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# predictions=[]

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# todo:
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
# Start Tensorflow session
# with tf.Session(config=config) as sess:
sess = tf.Session(config=config)

# Initialize all variables
# sess.run(init_op)
# sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

# coord=tf.train.Coordinator()
# threads=tf.train.start_queue_runners(coord=coord,sess=sess)

# Start the queue runners.
# tf.train.start_queue_runners(sess=sess,)

# Add the model graph to TensorBoard
writer.add_graph(sess.graph)

# Load the pretrained weights into the non-trainable layer
model.load_initial_weights(sess)

print("{} Start training...".format(datetime.now()))
print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                  filewriter_path))

# Loop over number of epochs
for epoch in range(num_epochs):

    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

    # Initialize iterator with the training dataset
    sess.run(training_init_op)

    for step in range(train_batches_per_epoch):

        # get next batch of data
        img_batch, label_batch = sess.run(next_batch)

        # And run the training op
        sess.run(train_op, feed_dict={x: img_batch,
                                      y: label_batch,
                                      keep_prob: dropout_rate})

        # Generate summary with the current batch of data and write to file
        if step % display_step == 0:
            s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.})

            writer.add_summary(s, epoch * train_batches_per_epoch + step)

    # Validate the model on the entire validation set
    print("{} Start validation".format(datetime.now()))
    sess.run(validation_init_op)
    test_acc = 0.
    test_count = 0
    for _ in range(val_batches_per_epoch):
        img_batch, label_batch = sess.run(next_batch)
        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                            y: label_batch,
                                            keep_prob: 1.})
        # pred = sess.run(softmax, feed_dict={x: img_batch})
        # predicted_label = pred.argmax(axis=1)
        # predictions.append(predicted_label[0])
        # auc_value= sess.run(auc, feed_dict={prediction_list:predictions,
        #                                y: label_batch,
        #                                keep_prob: 1.})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                   acc))
    # print('{} auc = {:.4f}'.format(datetime.now(), auc_value))
    print("{} Saving checkpoint of model...".format(datetime.now()))

    # save checkpoint of the model
    checkpoint_name = os.path.join(checkpoint_path,
                                   'model_epoch' + str(epoch + 1) + '.ckpt')
    save_path = saver.save(sess, checkpoint_name)

    print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                   checkpoint_name))

# coord.request_stop()
# coord.join(threads)
