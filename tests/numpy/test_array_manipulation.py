from unittest import TestCase

import numpy as np


class TestArrayManipulation(TestCase):
    def test_change_values(self):
        array = np.array([])
        assert array.ndim == 1
        assert array.size == 0
        with self.assertRaises(IndexError):
            array[0] = 0

        array = np.empty((2, 2))
        assert array.size == 4
        assert array.ndim == 2

        saved_value = array[0][0]
        array[0][0] = -(saved_value) + 2
        assert array[0][0] != saved_value

    def test_remove_values(self):
        # fmt: off
        array = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]
        ])
        assert array.size == 12
        assert array.ndim == 2
        assert array.shape == (4, 3)

        assert np.array_equal(
            np.delete(array, 0, 0),
            np.array([
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11]
            ])
        )

        assert np.array_equal(
            np.delete(array, 1, 0),
            np.array([
                [1, 2, 3],
                [6, 7, 8],
                [9, 10, 11]
            ])
        )

        assert np.array_equal(
            np.delete(array, [0, 2], 1),
            np.array([
                [2],
                [4],
                [7],
                [10]
            ])
        )

        assert np.array_equal(
            np.delete(array, -1, 1),
            np.array([
                [1, 2],
                [3, 4],
                [6, 7],
                [9, 10]
            ])
        )
        # fmt: on

    def test_append_values(self):
        array = np.array([[0, 1], [2, 3]])
        assert np.array_equal(
            np.append(array, [[4, 5]], axis=0), np.array([[0, 1], [2, 3], [4, 5]])
        )
        assert np.array_equal(
            np.append(array, [[4], [5]], axis=1), np.array([[0, 1, 4], [2, 3, 5]])
        )

    def test_insert(self):
        arange = np.arange(0, 6)
        array = np.array([arange, arange])
        assert array.shape == (2, 6)

        assert np.array_equal(
            np.insert(array, 0, 10, axis=0),
            np.array([[10, 10, 10, 10, 10, 10], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
        )

        assert np.array_equal(
            np.insert(array, 0, 10, axis=1),
            np.array([[10, 0, 1, 2, 3, 4, 5], [10, 0, 1, 2, 3, 4, 5]]),
        )

    def test_concat(self):
        a1 = np.array([np.arange(0, 3), np.arange(12, 15)])
        a2 = np.array([np.arange(20, 23)])
        assert np.array_equal(
            np.concatenate((a1, a2), axis=0), np.array([[0, 1, 2], [12, 13, 14], [20, 21, 22]])
        )

        a3 = np.array([[50], [100]])
        assert np.array_equal(
            np.concatenate((a1, a3), axis=1), np.array([[0, 1, 2, 50], [12, 13, 14, 100]])
        )

    def test_fill_diagonal(self):
        array = np.zeros((3, 3, 3))
        np.fill_diagonal(array, [50, 100])  # This is editing the original array

        assert np.array_equal(
            array,
            np.array(
                [
                    [[50, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 100, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 50]],
                ]
            ),
        )
