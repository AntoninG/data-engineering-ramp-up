from unittest import TestCase

import numpy as np
import pandas as pd


class TestPandasReshaping(TestCase):
    def setUp(self) -> None:
        super().setUp()
        arrays = [
            ["bar", "bar", "baz", "qux"],
            ["one", "two", "one", "two"],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        self.df = pd.DataFrame(
            [
                [-1.236148, 0.894879],
                [-0.211685, 0.179132],
                [1.235436, -1.545105],
                [1.387750, -1.581706],
            ],
            index=index,
            columns=["A", "B"],
        )

    def test_stack(self):
        expected_stacked_index = pd.MultiIndex.from_arrays(
            [
                ["bar", "bar", "bar", "bar", "baz", "baz", "qux", "qux"],
                ["one", "one", "two", "two", "one", "one", "two", "two"],
                ["A", "B", "A", "B", "A", "B", "A", "B"],
            ],
            names=["first", "second", None],
        )
        stacked = self.df.stack(future_stack=True)

        assert stacked.index.equals(expected_stacked_index)
        assert stacked.equals(
            pd.Series(
                [
                    -1.236148,
                    0.894879,
                    -0.211685,
                    0.179132,
                    1.235436,
                    -1.545105,
                    1.387750,
                    -1.581706,
                ],
                index=expected_stacked_index,
            )
        )

    def test_unstack(self):
        unstacked = self.df.unstack(1)
        expected_unstacked_index = pd.Index(["bar", "baz", "qux"], name="first")
        expected_unstacked_columns = pd.MultiIndex.from_arrays(
            [["A", "A", "B", "B"], ["one", "two", "one", "two"]], names=[None, "second"]
        )

        assert unstacked.index.equals(expected_unstacked_index)
        assert unstacked.columns.equals(expected_unstacked_columns)
        assert unstacked.equals(
            pd.DataFrame(
                [
                    [-1.236148, -0.211685, 0.894879, 0.179132],
                    [1.235436, np.nan, -1.545105, np.nan],
                    [np.nan, 1.387750, np.nan, -1.581706],
                ],
                index=expected_unstacked_index,
                columns=expected_unstacked_columns,
            )
        )

    def test_pivot(self):
        df = pd.DataFrame(
            {
                "A": ["A", "B", "C"] * 2,
                "B": ["one", "two"] * 3,
                "C": ["foo", "bar", "baz"] * 2,
                "D": np.random.randn(6),
            }
        )

        pivoted = pd.pivot_table(df, values="D", columns=["A", "C"], index="B")
        expected_index = pd.Index(["one", "two"], name="B")
        expected_columns = pd.MultiIndex.from_arrays(
            [["A", "B", "C"], ["foo", "bar", "baz"]], names=["A", "C"]
        )

        assert pivoted.index.equals(expected_index)
        assert pivoted.columns.equals(expected_columns)
