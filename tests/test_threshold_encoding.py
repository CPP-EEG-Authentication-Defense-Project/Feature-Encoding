import unittest
import numpy as np

from feature_encoding import threshold


class ThresholdEncodingTestCase(unittest.TestCase):
    def test_binary_encode_vector(self):
        test_data = np.random.rand(10)
        threshold_value = test_data.mean()
        encoder = threshold.ThresholdBinaryEncoder(threshold_value)

        binary_string = encoder.encode(test_data)

        self.assertIsInstance(binary_string, str)
        self.assertEqual(len(binary_string), len(test_data))

    def test_encode_vector_via_threshold(self):
        test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        threshold_value = 5
        encoder = threshold.ThresholdBinaryEncoder(threshold_value)
        expected_string = '0000011111'

        binary_string = encoder.encode(test_data)

        self.assertEqual(binary_string, expected_string)
