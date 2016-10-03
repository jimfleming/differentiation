from __future__ import print_function
from __future__ import division

import struct
import numpy as np

from utils import one_hot, shuffle

def read_labels(path):
    # [offset] [type]          [value]          [description]
    # 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    # 0004     32 bit integer  60000            number of items
    # 0008     unsigned byte   ??               label
    # 0009     unsigned byte   ??               label
    # .........
    # xxxx     unsigned byte   ??               label
    # The labels values are 0 to 9.
    with open(path, 'rb') as f:
        magic = struct.unpack('>i', f.read(4))[0]
        assert magic == 2049, 'magic number mismatch'
        num_labels = struct.unpack('>i', f.read(4))[0]
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

def read_images(path):
    # [offset] [type]          [value]          [description]
    # 0000     32 bit integer  0x00000803(2051) magic number
    # 0004     32 bit integer  60000            number of images
    # 0008     32 bit integer  28               number of rows
    # 0012     32 bit integer  28               number of columns
    # 0016     unsigned byte   ??               pixel
    # 0017     unsigned byte   ??               pixel
    # .........
    # xxxx     unsigned byte   ??               pixel
    with open(path, 'rb') as f:
        magic = struct.unpack('>i', f.read(4))[0]
        assert magic == 2051, 'magic number mismatch'
        num_images = struct.unpack('>i', f.read(4))[0]
        num_rows = struct.unpack('>i', f.read(4))[0]
        num_cols = struct.unpack('>i', f.read(4))[0]

        buf = f.read(num_images * num_rows * num_cols)
        images = np.frombuffer(buf, dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols, 1)
        return images

class Dataset(object):
    def __init__(self, images_path, labels_path, batch_size):
        self.images = read_images(images_path).astype(np.float32)
        self.images = self.images / 255.0 # normalize b/w 0..1
        self.images = self.images - np.mean(self.images) # mean subtraction
        self.images = self.images.reshape((-1, 784))

        self.labels = read_labels(labels_path)
        self.labels = one_hot(self.labels, 10)

        self.num_examples = len(self.labels)

        self.batch_size = batch_size
        self.batch_index = 0

        if self.batch_size:
            self.num_batches = self.num_examples // self.batch_size
        else:
            self.num_batches = 1

        self.images, self.labels = shuffle(self.images, self.labels)

    def next_batch(self):
        batch_start = self.batch_index * self.batch_size
        batch_end = batch_start + self.batch_size

        X = self.images[batch_start:batch_end]
        y = self.labels[batch_start:batch_end]

        self.batch_index += 1

        if self.batch_index > self.num_batches - 1:
            self.images, self.labels = shuffle(self.images, self.labels)
            self.batch_index = 0

        return X, y

    def batch_iter(self):
        for _ in range(self.num_batches):
            X, y = self.next_batch()
            yield X, y
