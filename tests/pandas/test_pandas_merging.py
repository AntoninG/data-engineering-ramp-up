from unittest import TestCase

import numpy as np
import pandas as pd


class TestPandasMerging(TestCase):
    def test_concat_series(self):
        s1 = pd.Series([1, 2, 3, 4, 5])
        s2 = pd.Series([11, 12, 13, 14, 15])

        concatenated = pd.concat([s1, s2], axis="index", ignore_index=True)
        assert isinstance(concatenated, pd.Series)
        assert concatenated.equals(pd.Series([1, 2, 3, 4, 5, 11, 12, 13, 14, 15]))

        concaneted_col_axis = pd.concat([s1, s2], axis="columns", ignore_index=True)
        assert isinstance(concaneted_col_axis, pd.DataFrame)
        assert concaneted_col_axis.equals(
            pd.DataFrame({0: [1, 2, 3, 4, 5], 1: [11, 12, 13, 14, 15]})
        )

    def test_concat_df(self):
        df1 = pd.DataFrame({"A": [0, 1, 2]})
        df2 = pd.DataFrame({"B": [10, 11, 12]})

        assert pd.concat([df1, df2], axis="index", ignore_index=True).equals(
            pd.DataFrame(
                {"A": [0, 1, 2, np.nan, np.nan, np.nan], "B": [np.nan, np.nan, np.nan, 10, 11, 12]}
            )
        )

        # ignore_index=False because we don't concat rows here, we want to keep col names
        assert pd.concat([df1, df2], axis="columns", ignore_index=False).equals(
            pd.DataFrame({"A": [0, 1, 2], "B": [10, 11, 12]})
        )

    def test_concat_series_df(self):
        series = pd.Series([0, 1, 2])
        dataframe = pd.DataFrame({"A": [0, 1, 2], "B": [10, 11, 12]})

        assert pd.concat([series, dataframe], axis="index", ignore_index=True).equals(
            pd.DataFrame(
                {
                    0: [0, 1, 2, np.nan, np.nan, np.nan],
                    "A": [np.nan, np.nan, np.nan, 0, 1, 2],
                    "B": [np.nan, np.nan, np.nan, 10, 11, 12],
                }
            )
        )

        assert pd.concat([series, dataframe], axis="columns").equals(
            pd.DataFrame({0: [0, 1, 2], "A": [0, 1, 2], "B": [10, 11, 12]})
        )
        assert pd.concat([dataframe, series], axis="columns").equals(
            pd.DataFrame({"A": [0, 1, 2], "B": [10, 11, 12], 0: [0, 1, 2]})
        )

    def test_join(self):
        table1 = pd.DataFrame({"name": ["foo", "bar", "baz"], "age": [28, 30, 65]})
        table2 = pd.DataFrame({"name": ["foo", "bar"], "weigth": [65, 92]})

        assert pd.merge(table1, table2, on="name").equals(
            pd.DataFrame({"name": ["foo", "bar"], "age": [28, 30], "weigth": [65, 92]})
        )

        assert pd.merge(table1, table2, on="name", how="left").equals(
            pd.DataFrame(
                {"name": ["foo", "bar", "baz"], "age": [28, 30, 65], "weigth": [65, 92, np.nan]}
            )
        )
