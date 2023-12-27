from unittest import TestCase

import pandas as pd


class TestPandasGrouping(TestCase):
    def test_groupby(self):
        df = pd.DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar"],
                "B": ["one", "two", "three", "one"],
                "C": [1.346061, 1.511763, 1.627081, -0.990582],
                "D": [-1.577585, 0.396823, -0.105381, -0.532532],
            }
        )

        grouped = df.groupby("A")
        assert grouped["C"].sum().equals(pd.Series([0.521181, 2.973142], index=["bar", "foo"]))
        assert grouped["C"].sum().sum() == 3.494323

        assert grouped["D"].sum().sum() == -1.818675

        assert grouped[["C", "D"]].sum().sum().sum() == (3.494323 + -1.818675)

        expected_index = pd.MultiIndex.from_arrays(
            [["bar", "bar", "foo", "foo"], ["one", "two", "one", "three"]], names=["A", "B"]
        )
        assert (
            df.groupby(["A", "B"])
            .sum()
            .equals(
                pd.DataFrame(
                    {
                        "C": [-0.990582, 1.511763, 1.346061, 1.627081],
                        "D": [-0.532532, 0.396823, -1.577585, -0.105381],
                    },
                    index=expected_index,
                )
            )
        )
