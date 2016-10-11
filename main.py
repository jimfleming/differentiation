from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np; np.random.seed(67)

from tqdm import trange, tqdm

from graph import Graph
from model import Model
from mnist import Dataset
from session import Session

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

def main():
    graph = Graph()
    sess = Session(graph)

    dataset_train = Dataset('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte', BATCH_SIZE)
    dataset_test = Dataset('data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte', None)

    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    # y = np.array([[0, 0, 0, 1]], dtype=np.float32) # AND/MIN
    # y = np.array([[0, 1, 1, 1]], dtype=np.float32) # OR/MAX
    # y = np.array([[0, 1, 1, 0]], dtype=np.float32) # XOR

    model = Model(LEARNING_RATE, BATCH_SIZE, graph)
    print('Training model with {} parameters'.format(model.size))

    with trange(NUM_EPOCHS) as pbar_epoch:
        for epoch in pbar_epoch:
            with tqdm(dataset_train.batch_iter(), total=dataset_train.num_batches, leave=False) as pbar_batch:
                batch_losses = []
                batch_accuracies = []

                for X, y in pbar_batch:
                    _, loss, accuracy = sess.run([model.update_op, model.loss, model.accuracy], feed_dict={
                        model.X: X,
                        model.y: y,
                    })

                    batch_losses.append(loss)
                    batch_accuracies.append(accuracy)

                    train_loss = np.mean(batch_losses)
                    train_accuracy = np.mean(batch_accuracies)

                    pbar_batch.set_description('loss: {:.8f}, accuracy: {:.2f}'.format(train_loss, train_accuracy))

            pbar_epoch.set_description('loss: {:.8f}, accuracy: {:.2f}'.format(train_loss, train_accuracy))

    epoch_losses = []
    epoch_accuracies = []
    for X, y in dataset_train.batch_iter():
        loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict={
            model.X: X,
            model.y: y,
        })
        epoch_losses.append(loss)
        epoch_accuracies.append(accuracy)
    test_loss = np.mean(epoch_losses)
    test_accuracy = np.mean(epoch_accuracies)
    print('loss: {:.8f}, accuracy: {:.2f}'.format(test_loss, test_accuracy))

if __name__ == '__main__':
    main()
