from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

from utils import reduced_shape

class OpsTestCase(unittest.TestCase):

    def test_reduced_shape(self):
        input_shape = [2, 3, 5, 7]
        axes = [1, 2]

        output_shape_kept_dims = reduced_shape(input_shape, axes, keep_dims=True)
        self.assertEqual(output_shape_kept_dims, [2, 1, 1, 7])

        output_shape = reduced_shape(input_shape, axes, keep_dims=False)
        self.assertEqual(output_shape, [2, 7])

        output_shape = reduced_shape(input_shape, None, keep_dims=False)
        self.assertEqual(output_shape, [])
