from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

from graph import Graph
from session import Session

class OpsTestCase(unittest.TestCase):

    def test_add(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()
        c = a + b

        sess = Session(graph)

        a_, b_, c_ = sess.run([a, b, c], feed_dict={a: 2, b: 1})

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 1)
        self.assertEqual(c_, 3)

    def test_sub(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()
        c = a - b

        sess = Session(graph)

        a_, b_, c_ = sess.run([a, b, c], feed_dict={a: 2, b: 3})

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 3)
        self.assertEqual(c_, -1)

    def test_mul(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()
        c = a * b

        sess = Session(graph)

        a_, b_, c_ = sess.run([a, b, c], feed_dict={a: 2, b: 3})

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 3)
        self.assertEqual(c_, 6)

    def test_div(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()
        c = a / b

        sess = Session(graph)

        a_, b_, c_ = sess.run([a, b, c], feed_dict={a: 6, b: 2})

        self.assertEqual(a_, 6)
        self.assertEqual(b_, 2)
        self.assertEqual(c_, 3)

    def test_square(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.square(a)

        sess = Session(graph)

        a_, b_ = sess.run([a, b], feed_dict={a: 3})

        self.assertEqual(a_, 3)
        self.assertEqual(b_, 9)

    def test_neg(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.neg(a)

        sess = Session(graph)

        a_, b_ = sess.run([a, b], feed_dict={a: 1})

        self.assertEqual(a_, 1)
        self.assertEqual(b_, -1)

    def test_sigmoid(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.sigmoid(a)

        sess = Session(graph)

        a_, b_ = sess.run([a, b], feed_dict={a: 1})

        self.assertEqual(a_, 1)
        self.assertAlmostEqual(b_, 0.731058579)

    def test_dot(self):
        graph = Graph()

        a = graph.tensor(value=np.array([0, 1, 2, 3]).reshape((1, -1)))
        b = graph.tensor(value=np.array([0, 1, 2, 3]).reshape((-1, 1)))
        c = graph.dot(a, b)

        sess = Session(graph)

        c_, = sess.run([c])

        self.assertTrue(np.array_equal(c_, [[14]]))

    def test_transpose(self):
        graph = Graph()

        a = graph.tensor(value=np.array([[0, 1, 2, 3]]))
        b = graph.transpose(a)

        sess = Session(graph)

        b_, = sess.run([b])

        self.assertTrue(np.array_equal(b_, np.array([[0], [1], [2], [3]])))

    def test_mean(self):
        graph = Graph()

        a = graph.tensor(value=np.array([[0, 2, 4, 6]]))
        b = graph.mean(a)

        sess = Session(graph)

        b_, = sess.run([b])

        self.assertEqual(b_, 3)

    def test_assign(self):
        graph = Graph()

        a = graph.tensor(1)
        increment_op = graph.assign(a, a + 1)

        sess = Session(graph)

        a0, = sess.run([a])
        sess.run([increment_op])
        a1, = sess.run([a])

        self.assertEqual(a0, 1)
        self.assertEqual(a1, 2)

    def test_assign_add(self):
        graph = Graph()

        a = graph.tensor(1)
        increment_op = graph.assign_add(a, 1)

        sess = Session(graph)

        a0, = sess.run([a])
        sess.run([increment_op])
        a1, = sess.run([a])

        self.assertEqual(a0, 1)
        self.assertEqual(a1, 2)

    def test_assign_sub(self):
        graph = Graph()

        a = graph.tensor(1)
        increment_op = graph.assign_sub(a, 1)

        sess = Session(graph)

        a0, = sess.run([a])
        sess.run([increment_op])
        a1, = sess.run([a])

        self.assertEqual(a0, 1)
        self.assertEqual(a1, 0)

if __name__ == '__main__':
    unittest.main()
