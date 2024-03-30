import unittest
import numpy as np

from feature_encoding import gray


class GrayEncodingTestCase(unittest.TestCase):
    def test_binary_encode_vector(self):
        test_data = np.random.randint(low=0, high=10, size=(10,))
        encoder = gray.GrayCodeBinaryEncoder(4)

        binary_string = encoder.encode(test_data)

        self.assertIsInstance(binary_string, str)

    def test_encode_vector_via_codebook(self):
        test_data = np.array([1, 3, 5, 7])
        encoder = gray.GrayCodeBinaryEncoder(4)
        expected_output = ''.join(['0001', '0010', '0111', '0100'])

        binary_string = encoder.encode(test_data)

        self.assertEqual(binary_string, expected_output)
