from __future__ import print_function
from __future__ import division

import numpy as np

from graph import Graph
from session import Session

def main():
    graph = Graph()

    a = graph.tensor(name='a') # input: 2
    b = graph.tensor(name='b') # input: 1

    c = graph.add(a, b, name='c') # output: 3
    d = graph.add(b, 1, name='d') # output: 2
    e = graph.mul(c, d, name='e') # output: 6

    sess = Session(graph)
    a_, b_, c_, d_, e_ = sess.run([a, b, c, d, e], feed_dict={a: 2, b: 1})
    print('results a: {}, b: {}, c: {}, d: {}, e: {}'.format(a_, b_, c_, d_, e_))

    de_da, de_db = graph.gradients(e, [a, b])
    de_da_, de_db_ = sess.run([de_da, de_db], {a: 2, b: 1})
    print('gradients de/da: {}, de/db: {}'.format(de_da_, de_db_))

if __name__ == '__main__':
    main()
