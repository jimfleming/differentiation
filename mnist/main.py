from __future__ import print_function
from __future__ import division

import random; random.seed(67)
import numpy as np; np.random.seed(67)

from model import Model
from mnist import Dataset
from trainer import Trainer

def main():
    batch_size = 32
    num_epochs = 10

    dataset_train = Dataset('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte', batch_size)
    dataset_test = Dataset('data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte', None)

    model = Model(batch_size)

    trainer = Trainer(model)
    trainer.fit(dataset_train, num_epochs)

    loss, accuracy = trainer.evaluate(dataset_test)
    print('[test] loss: {:.8f}, accuracy: {:.2f})'.format(loss, accuracy))

if __name__ == '__main__':
    main()
