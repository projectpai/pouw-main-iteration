import numpy as np


def get_outliers(np_array):
    q1, q3 = np.percentile(np_array, [25, 75])
    iqr = q3 - q1
    low = q1 - (iqr * 1.5)
    up = q3 + (iqr * 1.5)
    outliers_indices = np.where((np_array > up) | (np_array < low))[0]
    return np_array[outliers_indices]


# caution: this function assumes that item is a member of np_array
def is_outlier(item, np_array):
    return item in get_outliers(np_array)


if __name__ == '__main__':
    S = np.array([1, 2, 2, 6, 7, 1, 1, 3, 1, 1])
    print(get_outliers(S))
    print(is_outlier(6, S))
    print(is_outlier(1, S))
