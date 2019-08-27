import tensorflow as tf
import numpy as np
import random

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class PreDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, iterator_size, kth_init_op, classifier_version,
                 depth_num):
        self.txt_file = txt_file
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.depth = depth_num
        self.k = kth_init_op
        self.s = iterator_size

        # retrieve the data from the text file
        # todo: 0=remainder, 1=pred_know, 2=pred_unknow, 3=classifier, 4=random
        if classifier_version == 0:
            pass
        elif classifier_version == 1:
            self._read_txt_file_v1()
        elif classifier_version == 2:
            self._read_txt_file_v2()
        elif classifier_version == 3:
            self._read_txt_file_v3()
        elif classifier_version == 4:
            self._read_txt_file_randomly()
        else:
            raise ValueError('Invalid version v{}'.format(classifier_version))

        # number of samples in the dataset
        self.data_size = len(self.img_paths)

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        data = tf.data.Dataset.from_tensor_slices((self.img_paths))

        if mode == 'testing':
            data = data.map(self._parse_function_prediction, num_parallel_calls=8).prefetch(
                buffer_size=100 * batch_size)
        # distinguish between train/infer/pre. when calling the parsing functions
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        iterator = data.make_initializable_iterator()
        self.iterator = iterator
        self.data = data

    # to prediction know, include label
    def _read_txt_file_v1(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(',')  # .csv
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

    # to prediction unknown, no label
    def _read_txt_file_v2(self):  # ,kth_init_op,iterator_size
        self.img_paths = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()[self.s * self.k:self.s * self.k + self.s]
            for line in lines:
                items = line[:-1]  # .csv
                self.img_paths.append(items)

    # classifier_v4.py,多批次启动
    def _read_txt_file_v3(self):
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()[self.s * self.k:self.s * self.k + self.s]
            for line in lines:
                items = line.split(',')  # .csv
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

    #  to make randomly sample
    def _read_txt_file_randomly(self):
        self.img_paths = []
        self.labels = []  # comment it when prediction
        self.flen = len(open(self.txt_file, 'r').readlines())
        # self.samlen=2000
        sample_file = open("./SampleFile/zpsample_all500", 'w')
        # self.samlen=2000
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            # read_txt_file.ipynb
            np.random.seed(10)
            a = np.random.randint(self.flen, size=self.s)
            for i in range(self.s):
                items = lines[a[i]].split(',')  # .csv
                sample_file.write(str(items) + '\n')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

        # with open(self.txt_file, 'r') as f:
        #     lines = f.readlines()
        #     #read_txt_file.ipynb
        #     for i in range(self.s):
        #         random.seed(10)
        #         items = lines[random.randint(0,self.flen-1)].split(',')  # .csv
        #         self.img_paths.append(items[0])
        #         self.labels.append(int(items[1]))  # comment it when prediction

    def _parse_function_prediction(self, filename):
        """Input parser for samples of the prediction set."""

        # load and preprocess the input file
        height = 227
        width = 227
        depth = self.depth  # todo: parsing shape
        image_bytes = height * width * depth * 4

        img_string = tf.read_file(filename)
        bytes = tf.decode_raw(img_string, out_type=tf.float32)
        img = tf.reshape(bytes, [height, width, depth])

        return img
