from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np; np.random.seed(67)

from tqdm import trange

from graph import Graph
from model import Model
from session import Session

NUM_EPOCHS = 10000

def main():
    graph = Graph()
    session = Session(graph)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0, 0, 0, 1]], dtype=np.float32) # AND/MIN
    y = np.array([[0, 1, 1, 1]], dtype=np.float32) # OR/MAX
    y = np.array([[0, 1, 1, 0]], dtype=np.float32) # XOR

    model = Model(graph)

    print('Training model with {} parameters'.format(model.size))
    with trange(NUM_EPOCHS) as pbar:
        for epoch in pbar:
            _, loss = session.run([model.update_op, model.loss], feed_dict={
                model.X: X,
                model.y: y,
            })
            pbar.set_description('{:.5f}'.format(loss))

if __name__ == '__main__':
    main()
