from unittest import TestCase

import pandas as pd


class TestPandasBasicOperations(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.matrix = pd.DataFrame({"A": [1, 2, 3, 4], "B": [21, 23, 87, 12]})

    def test_mean(self):
        assert self.matrix.mean().equals(self.matrix.mean(axis="index"))
        assert self.matrix.mean().equals(pd.Series([2.5, 35.75], index=["A", "B"]))

        assert self.matrix.mean(axis="columns").equals(
            pd.Series([11, 12.5, 45, 8], index=[0, 1, 2, 3])
        )

    def test_abs(self):
        assert pd.Series([-0.333, 0.666]).abs().equals(pd.Series([0.333, 0.666]))

    def test_add(self):
        assert self.matrix.add(10).equals(
            pd.DataFrame({"A": [11, 12, 13, 14], "B": [31, 33, 97, 22]})
        )
        assert (self.matrix + 10).equals(
            pd.DataFrame({"A": [11, 12, 13, 14], "B": [31, 33, 97, 22]})
        )

    def test_sub(self):
        assert self.matrix.sub(1).equals(pd.DataFrame({"A": [0, 1, 2, 3], "B": [20, 22, 86, 11]}))
        assert (self.matrix - 1).equals(pd.DataFrame({"A": [0, 1, 2, 3], "B": [20, 22, 86, 11]}))

    def test_multiply(self):
        multiplied = self.matrix * 100
        assert multiplied.equals(
            pd.DataFrame({"A": [100, 200, 300, 400], "B": [2100, 2300, 8700, 1200]})
        )

    def test_transform(self):
        self.matrix.transform(lambda x: x * x).equals(
            pd.DataFrame({"A": [1, 4, 9, 16], "B": [441, 529, 7569, 144]})
        )

    def test_aggregate(self):
        agg = self.matrix.aggregate(lambda row_values: row_values.sum(), axis="columns")
        assert agg.equals(pd.Series([22, 25, 90, 16], index=[0, 1, 2, 3]))
        assert agg.sort_values(ascending=False).equals(
            pd.Series([90, 25, 22, 16], index=[2, 1, 0, 3])
        )

    def test_values_count(self):
        assert self.matrix.value_counts().tolist() == [1, 1, 1, 1]

    def test_string_methods(self):
        series = pd.Series(["A", "B", "C", "ZZ"]).transform(lambda string: f"{string}{string}")

        assert series.str.lower().equals(pd.Series(["aa", "bb", "cc", "zzzz"]))

        with self.assertRaises(AttributeError):
            # AttributeError: Can only use .str accessor with string values!. Did you mean: 'std'?
            pd.Series([1, 2, 3]).str.lower()
