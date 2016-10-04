from __future__ import print_function
from __future__ import division

import numpy as np

from graph import Graph
from session import Session

def main():
    graph = Graph()

    a = graph.tensor(name='a') # input: 2
    b = graph.tensor(name='b') # input: 1

    c = a + b # output: 3
    d = b + 1 # output: 2
    e = c * d # output: 6

    sess = Session(graph)
    result = sess.run(e, feed_dict={
        a: 2,
        b: 1,
    })
    print(result) # result: 6

if __name__ == '__main__':
    main()
