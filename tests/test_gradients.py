from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from ..graph import Graph
from ..session import Session

class GradientsTestCase(unittest.TestCase):

    def test_symbolic(self):
        # create a graph to hold computation
        graph = Graph()

        # create graph nodes representing a function
        a = graph.tensor() # input: 2
        b = graph.tensor() # input: 1

        c = a + b # output: 3
        d = b + 1 # output: 2
        e = c * d # output: 6

        # create graph nodes representing gradient of function w.r.t. inputs
        de_da, de_db = graph.gradients(e, [a, b])

        # create a session to perform computation
        sess = Session(graph)

        # evaluate graph nodes with the session
        a_, b_, c_, d_, e_, de_da_, de_db_ = sess.run([
            a,
            b,
            c,
            d,
            e,
            de_da,
            de_db,
        ], feed_dict={
            a: 2,
            b: 1,
        })

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 1)
        self.assertEqual(c_, 3)
        self.assertEqual(d_, 2)
        self.assertEqual(e_, 6)
        self.assertEqual(de_da_, 2)
        self.assertEqual(de_db_, 5)

if __name__ == '__main__':
    unittest.main()
