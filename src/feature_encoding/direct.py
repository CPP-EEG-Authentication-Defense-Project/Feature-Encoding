import struct
import numpy as np

from . import base


class DirectBinaryEncoder(base.BinaryEncoder):
    """
    Converts feature vectors into direct binary representations, either using their float number representation
    or by first rounding each element to a whole number and using the integer representation.
    """
    def __init__(self, round_to_whole=False):
        self.round_to_whole = round_to_whole

    def convert_to_binary(self, data: np.ndarray) -> str:
        if self.round_to_whole:
            rounded_data = np.round(data)
            ''.join([np.binary_repr(element) for element in rounded_data])
        return ''.join([self._convert_float_to_binary(element) for element in data])

    @staticmethod
    def _convert_float_to_binary(number: float) -> str:
        """
        Converts the given float value into its binary representation.

        Originally taken from https://stackoverflow.com/a/16444778/13261549.

        :param number: The number to convert to binary.
        :return: The binary representation of the number.
        """
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', number))

