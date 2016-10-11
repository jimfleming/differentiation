from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

from graph import Graph
from session import Session

class GradientsTestCase(unittest.TestCase):

    def test_add_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()
        c = a + b

        sess = Session(graph)

        grad_a, grad_b = graph.gradients(c, [a, b])
        grad_a_, grad_b_ = sess.run([grad_a, grad_b], feed_dict={a: 2, b: 1})

        self.assertEqual(grad_a_, 1)
        self.assertEqual(grad_b_, 1)

    def test_sub_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()
        c = a - b

        sess = Session(graph)

        grad_a, grad_b = graph.gradients(c, [a, b])
        grad_a_, grad_b_ = sess.run([grad_a, grad_b], feed_dict={a: 2, b: 1})

        self.assertEqual(grad_a_, 1)
        self.assertEqual(grad_b_, -1)

    def test_mul_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()
        c = a * b

        sess = Session(graph)

        grad_a, grad_b = graph.gradients(c, [a, b])
        grad_a_, grad_b_ = sess.run([grad_a, grad_b], feed_dict={a: 2, b: 3})

        self.assertEqual(grad_a_, 3)
        self.assertEqual(grad_b_, 2)

    def test_div_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()
        c = a / b

        sess = Session(graph)

        grad_a, grad_b = graph.gradients(c, [a, b])
        grad_a_, grad_b_ = sess.run([grad_a, grad_b], feed_dict={a: 2, b: 3})

        self.assertAlmostEqual(grad_a_, 0.3333333)
        self.assertAlmostEqual(grad_b_, -0.2222222)

    def test_square_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.square(a)

        sess = Session(graph)

        grad, = graph.gradients(b, [a])
        grad_, = sess.run([grad], feed_dict={a: 6})

        self.assertEqual(grad_, 12)

    def test_power_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.power(a, 3)

        sess = Session(graph)

        grad, = graph.gradients(b, [a])
        grad_, = sess.run([grad], feed_dict={a: 6})

        self.assertEqual(grad_, 108)

    def test_log_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.log(a)

        sess = Session(graph)

        grad, = graph.gradients(b, [a])
        grad_, = sess.run([grad], feed_dict={a: 6})

        self.assertAlmostEqual(grad_, 1/6)

    def test_sigmoid_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.sigmoid(a)

        sess = Session(graph)

        grad, = graph.gradients(b, [a])
        grad_, = sess.run([grad], feed_dict={a: 1})

        self.assertAlmostEqual(grad_, 0.19661193)

    def test_neg_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = -a

        sess = Session(graph)

        grad, = graph.gradients(b, [a])
        grad_, = sess.run([grad], feed_dict={a: 1})

        self.assertEqual(grad_, -1)

    def test_dot_grad(self):
        graph = Graph()

        a = graph.tensor(value=np.array([0, 1, 2, 3]).reshape((1, -1)))
        b = graph.tensor(value=np.array([0, 1, 2, 3]).reshape((-1, 1)))
        c = graph.dot(a, b)

        sess = Session(graph)

        grad_a, grad_b, = graph.gradients(c, [a, b])
        grad_a_, grad_b_ = sess.run([grad_a, grad_b])

        self.assertTrue(np.array_equal(grad_a_, np.array([[0, 1, 2, 3]])))
        self.assertTrue(np.array_equal(grad_b_, np.array([[0], [1], [2], [3]])))

    def test_transpose_grad(self):
        graph = Graph()

        a = graph.tensor(value=np.array([[0, 1, 2, 3]]))
        b = graph.transpose(a)

        sess = Session(graph)

        grad, = graph.gradients(b, [a])
        grad_, = sess.run([grad])

        self.assertTrue(np.array_equal(grad_, np.array([[1, 1, 1, 1]])))

    def test_mean_grad(self):
        graph = Graph()

        a = graph.tensor(value=np.array([[0, 2, 4, 6]]))
        b = graph.mean(a)

        sess = Session(graph)

        grad, = graph.gradients(b, [a])
        grad_, = sess.run([grad])

        self.assertTrue(np.array_equal(grad_, np.array([[0.25, 0.25, 0.25, 0.25]])))

    def test_sum_grad(self):
        graph = Graph()

        a = graph.tensor(value=np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]))
        b = graph.sum(a, axis=1)

        sess = Session(graph)

        grad, = graph.gradients(b, [a])
        grad_, = sess.run([grad])

        self.assertTrue(np.array_equal(grad_, np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ])))

    def test_expression_grad(self):
        graph = Graph()

        a = graph.tensor()
        b = graph.tensor()

        c = a + b
        d = b + 1
        e = c * d

        de_da, de_db = graph.gradients(e, [a, b])

        sess = Session(graph)

        a_, b_, c_, d_, e_, de_da_, de_db_ = sess.run([a, b, c, d, e, de_da, de_db], feed_dict={a: 2, b: 1})

        self.assertEqual(a_, 2)
        self.assertEqual(b_, 1)
        self.assertEqual(c_, 3)
        self.assertEqual(d_, 2)
        self.assertEqual(e_, 6)
        self.assertEqual(de_da_, 2)
        self.assertEqual(de_db_, 5)

if __name__ == '__main__':
    unittest.main()
