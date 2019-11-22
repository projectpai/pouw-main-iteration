import os

import numpy as np
from livestats import livestats


# class using LiveStats
class LiveMedian:
    def __init__(self):
        self._ls = livestats.LiveStats([0.5])  # we are only interested in the median

    def add(self, item):
        self._ls.add(item)

    def get_median(self):
        return self._ls.quantiles()[0][1] if self._ls.count > 0 else float('nan')


def get_network_time_to_pick_next_message(omega, gradients_sizes, features_file_path, labels_file_path):
    # calculate model size
    model_size = sum(gradients_sizes)
    # calculate batch size as the sum of the features and labels
    stat_info_features = os.stat(features_file_path)
    stat_info_labels = os.stat(labels_file_path)
    batch_size_on_disk = stat_info_features.st_size + stat_info_labels.st_size
    # calculate network allowed time to pick the next message
    return omega * model_size * batch_size_on_disk


# method using Numpy
def np_calculate_median(np_array):
    return np.median(np_array)


if __name__ == '__main__':
    S = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    numpy_median = np_calculate_median(np.array(S))

    lm = LiveMedian()
    for i in S:
        lm.add(i)  # we add items as we go, in an online fashion

    live_stats_median = lm.get_median()

    assert numpy_median == live_stats_median

    print("NumPy median: ", numpy_median)
    print("LiveStats median: ", live_stats_median)
