import unittest
import numpy as np

from feature_encoding import direct


class DirectEncodingTestCase(unittest.TestCase):
    def test_binary_encode_vector(self):
        test_data = np.random.rand(10)
        encoder = direct.DirectBinaryEncoder()

        binary_string = encoder.encode(test_data)

        self.assertIsInstance(binary_string, str)

    def test_encode_vector_with_rounding(self):
        test_data = np.array([2.2, 3.1, 4.3, 4.9])
        encoder = direct.DirectBinaryEncoder(round_to_whole=True)
        expected_output = ''.join(['10', '11', '100', '101'])

        binary_string = encoder.encode(test_data)

        self.assertEqual(binary_string, expected_output)

    def test_encode_float_vector(self):
        test_data = np.array([2.2, 3.1, 4.3, 4.9])
        encoder = direct.DirectBinaryEncoder(round_to_whole=False)
        expected_output = ''.join([
            '01000000000011001100110011001101',
            '01000000010001100110011001100110',
            '01000000100010011001100110011010',
            '01000000100111001100110011001101'
        ])

        binary_string = encoder.encode(test_data)

        self.assertEqual(binary_string, expected_output)
