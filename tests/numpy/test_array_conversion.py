from unittest import TestCase

import numpy as np


class TestArrayConversion(TestCase):
    def test_to_list(self):
        array = np.zeros((2, 2))
        assert array.tolist() == [[0, 0], [0, 0]]

    def test_to_set(self):
        # fmt: off
        array = np.array([
            [
                [1, 2]
            ],
            [
                [50, 100]
            ],
            [
                [10, 16]
            ]
        ])
        # fmt: on

        unique = np.unique(array)
        assert np.array_equal(unique, [1, 2, 10, 16, 50, 100])

        assert set(unique) == {1, 2, 10, 16, 50, 100}

    def test_reshape(self):
        array = np.arange(6)

        assert np.array_equal(array.reshape((3, 2)), [[0, 1], [2, 3], [4, 5]])
        assert np.array_equal(array.reshape((2, 3)), [[0, 1, 2], [3, 4, 5]])

        with self.assertRaises(ValueError):
            array.reshape((12, 2))
