from __future__ import print_function
from __future__ import division

import numpy as np

from tqdm import tqdm, trange
from utils import cross_entropy, label_accuracy

class Trainer(object):

    def __init__(self, model):
        self.model = model

    def fit(self, dataset, num_epochs):
        with trange(num_epochs) as pbar_epoch:
            epoch_losses = []
            epoch_accuracies = []
            for epoch in pbar_epoch:
                batch_losses = []
                batch_accuracies = []
                with tqdm(dataset.batch_iter(), total=dataset.num_batches, leave=False) as pbar_batch:
                    for X, y in pbar_batch:
                        y_ = self.model.forward(X)
                        self.model.backward(X, y)

                        loss = cross_entropy(y_, y)
                        # loss += self.model.regularization()

                        accuracy = label_accuracy(y_, y)

                        batch_losses.append(loss)
                        batch_accuracies.append(accuracy)

                        loss_mean = np.mean(batch_losses)
                        accuracy_mean = np.mean(batch_accuracies)
                        pbar_batch.set_description('loss: {:.8f}, accuracy: {:.2f}'.format(loss_mean, accuracy_mean))

                epoch_losses.append(loss_mean)
                epoch_accuracies.append(accuracy_mean)
                pbar_epoch.set_description('loss: {:.8f}, accuracy: {:.2f}'.format(loss_mean, accuracy_mean))

    def evaluate(self, dataset):
        X, y = dataset.images, dataset.labels
        y_ = self.model.forward(X)
        loss = cross_entropy(y_, y)
        accuracy = label_accuracy(y_, y)
        return loss, accuracy
