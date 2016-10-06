from __future__ import print_function
from __future__ import division

import unittest

from graph import Graph
from session import Session

class SymbolicTestCase(unittest.TestCase):

    def test_symbolic(self):
        graph = Graph()

        a = graph.tensor() # input: 2
        b = graph.tensor() # input: 1

        c = a + b # output: 3
        d = b + 1 # output: 2
        e = c * d # output: 6

        sess = Session(graph)
        a_, b_, c_, d_, e_ = sess.run([a, b, c, d, e], feed_dict={a: 2, b: 1})

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 1)
        self.assertEqual(c_, 3)
        self.assertEqual(d_, 2)
        self.assertEqual(e_, 6)

        de_da, de_db = graph.gradients(e, [a, b])
        de_da_, de_db_ = sess.run([de_da, de_db], {a: 2, b: 1})

        self.assertEqual(de_da_, 2)
        self.assertEqual(de_db_, 5)

if __name__ == '__main__':
    unittest.main()
