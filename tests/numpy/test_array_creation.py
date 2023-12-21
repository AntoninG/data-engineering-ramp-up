from unittest import TestCase

import numpy as np


class TestArrayCreation(TestCase):
    def test_basic_array(self):
        array = np.array([1, 2, 3])
        assert array.size == 3
        assert array.ndim == 1
        assert array[-1] == 3

    def test_ndarray(self):
        array = np.ndarray([2, 3, 6])
        assert array.size == 2 * 3 * 6
        assert array.ndim == 3
        assert array.shape == (2, 3, 6)

    def test_invalid_array_content_raises_error(self):
        with self.assertRaises(Exception):
            np.array([1, 2, 3], np.datetime64)

    def test_empty_array_is_not_empty(self):
        array = np.empty(2)
        assert array.size == 2
        assert array.ndim == 1

    def test_zero_only_array(self):
        array = np.zeros([2, 2])
        assert array.size == 2 * 2
        assert array.ndim == 2
        assert array[0][0] == 0

    def test_range(self):
        array = np.arange(start=2, stop=9, step=2, dtype=np.float64)
        assert array.size == 4
        assert array.ndim == 1
        assert array[0] == 2.0
        assert array[-1] == 8.0

    def test_linear_spaced_array(self):
        array = np.linspace(
            start=0,
            stop=100,
            num=3,
        )
        assert array.size == 3
        assert array.ndim == 1
        assert array[0] == 0
        assert array[-1] == 100

        array = np.linspace(start=0, stop=100, num=3, endpoint=False)
        assert array.size == 3
        assert array.ndim == 1
        assert array[0] == 0
        assert array[-1] != 100

    def test_dumb_dimension_raises_error(self):
        with self.assertRaises(ValueError):
            np.array([[[1, 2, 3], [3, 4, 5]], [[6, 7, 8], []]])
