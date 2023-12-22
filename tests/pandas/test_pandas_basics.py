from unittest import TestCase

import numpy as np
import pandas as pd


class TestPandasBasics(TestCase):
    def test_series_can_contains_mixed_types(self):
        array = pd.Series(["Any", "type", 1, 1.0, True])
        assert array.dtype == np.dtype(object)

    def test_series_single_type(self):
        array = pd.Series([1.0, 5.0])
        assert array.dtype == np.dtype("float64")

    def test_series_named_indexes(self):
        array = pd.Series([1.0, 5.0], index=["A", "B"])
        assert repr(array) == "A    1.0\nB    5.0\ndtype: float64"

    def test_dataframe_can_contains_mixed_types(self):
        array = pd.DataFrame([[1, 2, 3], ["A", "B", "C"]])
        assert array.dtypes.dtype == np.dtype(object)  # First column type is object because mixed

    def test_dataframe_named_columns(self):
        array = pd.DataFrame({"A": [0, 1], "Name": "Foo", "spend": 100_000.15})

        assert array.dtypes["A"] == np.dtype(int)
        assert array.dtypes["Name"] == np.dtype(object)  # Idk why it is not str
        assert array.dtypes["spend"] == np.dtype(float)

        assert array.size == 6  # Because array in 1 column makes 2 rows
        assert array["Name"][1] == "Foo"

    def test_view_dataframe(self):
        array = pd.DataFrame({"A": [0, 1, 2], "C": [9, 64, 78], "EE": 100})

        assert array.head(1).equals(pd.DataFrame({"A": [0], "C": [9], "EE": [100]}))
        assert array.tail(2).equals(
            pd.DataFrame({"A": {1: 1, 2: 2}, "C": {1: 64, 2: 78}, "EE": {1: 100, 2: 100}})
        )

        assert array.index.tolist() == [0, 1, 2]
        assert array.columns.tolist() == ["A", "C", "EE"]

    def test_sorting_values_dataframe(self):
        array = pd.DataFrame({"A": [2, 1, 0], "C": [9, 64, 78], "EE": 100})

        assert array.sort_values(by="A", ascending=True).equals(
            pd.DataFrame(
                {"A": {2: 0, 1: 1, 0: 2}, "C": {2: 78, 1: 64, 0: 9}, "EE": {2: 100, 1: 100, 0: 100}}
            )
        )

    def test_sorting_index_dataframe(self):
        array = pd.DataFrame({"A": [2, 1, 0], "C": [9, 64, 78], "EE": 100})

        assert array.sort_index(axis="columns", ascending=False).equals(
            pd.DataFrame({"EE": 100, "C": [9, 64, 78], "A": [2, 1, 0]})
        )

    def test_get_data_in_dataframe(self):
        array = pd.DataFrame({"A": [0, 1, 2], "C": [9, 64, 78], "EE": 100})

        assert array["A"].tolist() == [0, 1, 2]

        assert array[0:2].equals(pd.DataFrame({"A": [0, 1], "C": [9, 64], "EE": 100}))
        assert array[0:2].equals(array.head(2))

        assert array[1:2].equals(pd.DataFrame({"A": {1: 1}, "C": {1: 64}, "EE": {1: 100}}))

        assert array[["A", "EE"]].equals(pd.DataFrame({"A": [0, 1, 2], "EE": 100}))

        assert array.at[2, "EE"] == 100

        assert array.loc[[1], ["A", "EE"]].equals(pd.DataFrame({"A": {1: 1}, "EE": {1: 100}}))
        assert array.loc[1, ["A", "EE"]].equals(pd.Series([1, 100], ["A", "EE"]))

        assert array.iat[0, -1] == 100  # First row, last col

        assert array.iloc[1:2, [0, 2]].equals(pd.DataFrame({"A": {1: 1}, "EE": {1: 100}}))
