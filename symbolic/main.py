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
    results = sess.run([c, d, e], feed_dict={
        a: 2,
        b: 1,
    })
    print('results', results) # result: [3, 2, 6]

    a = 2
    b = 1
    c = a + b # = 3
    d = b + 1 # = 2
    e = c * d # = 6

    # identities
    dd_dc = 0
    dc_dd = 0
    db_da = 0

    da_da = 1
    db_db = 1
    dc_dc = 1
    dd_dd = 1

    de_de = 1

    de_dc = (dd_dc * c) + (dc_dc * d) # = d = 2
    de_dd = (dc_dd * d) + (dd_dd * c) # = c = 3

    dc_da = da_da + db_da # = 1
    dc_db = db_da + da_da # = 1

    dd_db = 1

    de_da = (dc_da * de_dc) # = 2
    de_db = (dc_db * de_dc) + (dd_db * de_dd) # = 5

    print(de_da, de_db) # [2, 5]

    # grad_a, grad_b = graph.gradients(e, [a, b], name='gradients')
    # gradients = sess.run([grad_a, grad_b], feed_dict={
    #     a: 2,
    #     b: 1,
    # })
    # print('gradients', gradients) # result: [[2, 5]]

if __name__ == '__main__':
    main()
