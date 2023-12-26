from unittest import TestCase

import pandas as pd


class TestPandasBrowsing(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.df = pd.DataFrame({"A": [0, 1], "name": ["foo", "bar"], "age": [28, 30]})

    def test_loop_series(self):
        series = pd.Series([0, 50], index=["A", "B"])

        assert list(series.items()) == [("A", 0), ("B", 50)]

    def test_loop_dataframe_items(self):
        for column, series in self.df.items():
            assert column in ["A", "name", "age"]
            assert isinstance(series, pd.Series)

            series_list = series.tolist()
            assert series_list in ([0, 1], ["foo", "bar"], [28, 30])

    def test_loop_dataframe_tuples(self):
        for pandas in self.df.itertuples():
            assert pandas.Index in [0, 1]
            assert pandas.A in [0, 1]
            assert pandas.name in ["foo", "bar"]
            assert pandas.age in [28, 30]

        for tu in self.df.itertuples(name=None, index=False):
            assert tu in ((0, "foo", 28), (1, "bar", 30))

    def test_loop_dataframe_rows(self):
        for index, series_row in self.df.iterrows():
            assert index in [0, 1]
            assert series_row.equals(
                pd.Series({"A": 0, "name": "foo", "age": 28})
            ) or series_row.equals(pd.Series({"A": 1, "name": "bar", "age": 30}))
