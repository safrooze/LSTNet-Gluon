from __future__ import print_function

from mxnet import gluon
import numpy as np


class TimeSeriesData(object):
    """
    Reads data from file and creates training and validation datasets
    """
    def __init__(self, file_path, window, horizon, train_ratio=0.8):
        """
        :param str file_path: path to the data file (e.g. electricity.txt)
        """
        # with open(file_path) as f:
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        data = self._normalize(data)
        train_data_len = int(len(data) * train_ratio)
        self.num_series = data.shape[1]
        if train_ratio > 0.0:
            self.train = TimeSeriesDataset(data[:train_data_len], window=window, horizon=horizon)
        if train_ratio < 1.0:
            self.val = TimeSeriesDataset(data[train_data_len:], window=window, horizon=horizon)

    def _normalize(self, data):
        """ Normalizes data by maximum value per row (i.e. per time series) and saves the scaling factor

        :param np.ndarray data: input data to be normalized
        :return: normalized data
        :rtype np.ndarray
        """
        self.scale = np.max(data, axis=0)
        return data / self.scale


class TimeSeriesDataset(gluon.data.Dataset):
    """
    Dataset that splits the data into a dense overlapping windows
    """
    def __init__(self, data, window, horizon, transform=None):
        """
        :param np.ndarray data: time-series data in TC layout (T: sequence len, C: channels)
        :param int window: context window size
        :param int horizon: prediction horizon
        :param function transform: data transformation function: fn(data, label)
        """
        super(TimeSeriesDataset, self).__init__()
        self._data = data
        self._window = window
        self._horizon = horizon
        self._transform = transform

    def __getitem__(self, idx):
        """
        :param int idx: index of the item
        :return: single item in 'TC' layout
        :rtype np.ndarray
        """
        assert idx < len(self)
        data = self._data[idx:idx + self._window]
        label = self._data[idx + self._window + self._horizon - 1]
        if self._transform is not None:
            return self._transform(data, label)
        return data, label

    def __len__(self):
        """
        :return: length of the dataset
        :rtype int
        """
        return len(self._data) - self._window - self._horizon


if __name__ == "__main__":
    """
    Run unit-test
    """
    dataset = TimeSeriesDataset(np.arange(0, 100).reshape(-1, 1).repeat(10, axis=1), window=10, horizon=5)
    assert len(dataset) == 85
    for i in range(len(dataset)):
        d, l = dataset[i]
        assert np.array_equal(d, np.arange(i, i + 10).reshape(-1, 1).repeat(10, axis=1))
        assert np.array_equal(l, np.array([i + 14]).repeat(10, axis=0))
    print("Unit-test success!")
