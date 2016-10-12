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

        a = graph.tensor(shape=())
        b = graph.tensor(shape=())
        c = a + b

        sess = Session(graph)

        a_, b_, c_ = sess.run([a, b, c], feed_dict={a: 2, b: 1})

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 1)
        self.assertEqual(c_, 3)

    def test_sub(self):
        graph = Graph()

        a = graph.tensor(shape=())
        b = graph.tensor(shape=())
        c = a - b

        sess = Session(graph)

        a_, b_, c_ = sess.run([a, b, c], feed_dict={a: 2, b: 3})

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 3)
        self.assertEqual(c_, -1)

    def test_mul(self):
        graph = Graph()

        a = graph.tensor(shape=())
        b = graph.tensor(shape=())
        c = a * b

        sess = Session(graph)

        a_, b_, c_ = sess.run([a, b, c], feed_dict={a: 2, b: 3})

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 3)
        self.assertEqual(c_, 6)

    def test_div(self):
        graph = Graph()

        a = graph.tensor(shape=())
        b = graph.tensor(shape=())
        c = a / b

        sess = Session(graph)

        a_, b_, c_ = sess.run([a, b, c], feed_dict={a: 6, b: 2})

        self.assertEqual(a_, 6)
        self.assertEqual(b_, 2)
        self.assertEqual(c_, 3)

    def test_square(self):
        graph = Graph()

        a = graph.tensor(shape=())
        b = graph.square(a)

        sess = Session(graph)

        a_, b_ = sess.run([a, b], feed_dict={a: 3})

        self.assertEqual(a_, 3)
        self.assertEqual(b_, 9)

    def test_power(self):
        graph = Graph()

        a = graph.tensor(shape=())
        b = graph.power(a, 3)

        sess = Session(graph)

        a_, b_ = sess.run([a, b], feed_dict={a: 3})

        self.assertEqual(a_, 3)
        self.assertEqual(b_, 27)

    def test_log(self):
        graph = Graph()

        a = graph.tensor(shape=())
        b = graph.log(a)

        sess = Session(graph)

        a_, b_ = sess.run([a, b], feed_dict={a: 3})

        self.assertEqual(a_, 3)
        self.assertAlmostEqual(b_, 1.098612289)

    def test_neg(self):
        graph = Graph()

        a = graph.tensor(shape=())
        b = graph.neg(a)

        sess = Session(graph)

        a_, b_ = sess.run([a, b], feed_dict={a: 1})

        self.assertEqual(a_, 1)
        self.assertEqual(b_, -1)

    def test_sigmoid(self):
        graph = Graph()

        a = graph.tensor(shape=())
        b = graph.sigmoid(a)

        sess = Session(graph)

        a_, b_ = sess.run([a, b], feed_dict={a: 1})

        self.assertEqual(a_, 1)
        self.assertAlmostEqual(b_, 0.731058579)

    def test_softmax(self):
        graph = Graph()

        a = graph.tensor(shape=(3,))
        b = graph.softmax(a)

        sess = Session(graph)

        b_, = sess.run([b], feed_dict={a: np.array([-3, 0, 3])})

        self.assertTrue(np.allclose(b_, [0.00235563, 0.04731416, 0.95033021]))

    def test_relu(self):
        graph = Graph()

        a = graph.tensor(shape=(3,))
        b = graph.relu(a)

        sess = Session(graph)

        b_, = sess.run([b], feed_dict={a: np.array([-3, 0, 3])})

        self.assertTrue(np.array_equal(b_, [0, 0, 3]))

    def test_where(self):
        graph = Graph()

        a = graph.tensor(shape=(3,))
        b = graph.where(a > 0, 1, 0)

        sess = Session(graph)

        b_, = sess.run([b], feed_dict={a: np.array([-3, 0, 3])})

        self.assertTrue(np.array_equal(b_, [0, 0, 1]))

    def test_equal(self):
        graph = Graph()

        a = graph.tensor(shape=(3,))
        b = graph.tensor(shape=(3,))
        c = graph.equal(a, b)

        sess = Session(graph)

        c_, = sess.run([c], feed_dict={
            a: np.array([-3, 0, 3]),
            b: np.array([-2, 1, 3]),
        })

        self.assertTrue(np.array_equal(c_, [0, 0, 1]))

    def test_argmax(self):
        graph = Graph()

        a = graph.tensor(shape=())
        c = graph.argmax(a)

        sess = Session(graph)

        c_, = sess.run([c], feed_dict={a: np.array([-3, 0, 3])})

        self.assertEqual(c_, 2)

    def test_greater(self):
        graph = Graph()

        a = graph.tensor(shape=(3,))
        b = a > 0

        sess = Session(graph)

        b_, = sess.run([b], feed_dict={a: np.array([-3, 0, 3])})

        self.assertTrue(np.array_equal(b_, [0, 0, 1]))

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

    def test_sum(self):
        graph = Graph()

        a = graph.tensor(value=np.array([[0, 1, 2, 3]]))
        b = graph.sum(a)

        sess = Session(graph)

        b_, = sess.run([b])

        self.assertEqual(b_, 6)

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
