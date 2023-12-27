from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd


class TestPandasDates(TestCase):
    def test_range(self):
        dates_index = pd.date_range(datetime.now(), periods=10, freq="2D")
        series = pd.Series(np.arange(10), index=dates_index)

        assert series.size == 10

        first_date: datetime = series.index[0]
        assert first_date.timetuple()[0:3] == datetime.now().timetuple()[0:3]

        last_date: datetime = series.index[-1]
        assert last_date.timetuple()[0:3] == (datetime.now() + timedelta(days=18)).timetuple()[0:3]

    def test_resample(self):
        dates_index = pd.date_range(datetime.now(), periods=10, freq="2D")
        series = pd.Series(np.arange(10), index=dates_index)

        resampled = series.resample("5min").size()
        assert resampled.size > 5000
        assert resampled.loc[lambda value: value > 0].size == 10
