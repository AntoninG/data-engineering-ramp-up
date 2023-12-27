from unittest import TestCase

import numpy as np
import pandas as pd


class TestPandasCategoricals(TestCase):
    def test_categoricals(self):
        df = pd.DataFrame(
            {"student": ["John", "Liv", "Elsa", "Vince", "Dick"] * 300, "rank": np.arange(5 * 300)}
        )

        size = df.size
        memory_usage = df.memory_usage().sum()
        df["student"] = df["student"].astype("category")

        assert df.memory_usage().sum() < memory_usage

        full_name_students = [
            "John 117",
            "Liv Tyler",
            "Elsa Tchoyn",
            "Vince Venturella",
            "Dick Dickers",
        ]
        df["student"] = df["student"].cat.rename_categories(full_name_students)

        assert df.size == size
