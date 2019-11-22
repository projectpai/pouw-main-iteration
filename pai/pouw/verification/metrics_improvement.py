import numpy as np


# Code inspired from the implementation here: https://www.swharden.com/wp/2008-11-17-linear-data-smoothing-in-python/


# check if list strictly decreasing
def strictly_decreasing(val_list):
    return all(x >= y for x, y in zip(val_list, val_list[1:]))


# check if list is strictly increasing
def strictly_increasing(val_list):
    return all(x <= y for x, y in zip(val_list, val_list[1:]))


# moving average smoothing
def smooth(val_list, degree=10):
    smoothed = [0] * (len(val_list) - degree + 1)

    for i in range(len(smoothed)):
        smoothed[i] = sum(val_list[i:i + degree]) / float(degree)

    return smoothed


# triangle method smoothing
def smooth_triangle(val_list, degree=5):
    weight = []

    window = degree * 2 - 1

    smoothed = [0.0] * (len(val_list) - window)

    for x in range(1, 2 * degree):
        weight.append(degree - abs(degree - x))

    w = np.array(weight)

    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(val_list[i:i + window]) * w) / float(sum(w))

    return smoothed


# Gaussian smoothing
def smooth_gaussian(val_list, degree=5):
    window = degree * 2 - 1

    weight = np.array([1.0] * window)

    weight_gauss = []

    for i in range(window):
        i = i - degree + 1
        gauss = 1 / (np.exp((4 * (i / float(window))) ** 2))
        weight_gauss.append(gauss)

    weight = np.array(weight_gauss) * weight
    smoothed = [0.0] * (len(val_list) - window)

    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(val_list[i:i + window]) * weight) / sum(weight)

    return smoothed


if __name__ == '__main__':
    S = [0.06, 0.05, 0.04, 0.09, 0.07, 0.06, 0.11, 0.08, 0.14]
    time_window = 5
    smooth_1 = smooth(S, time_window)
    smooth_2 = smooth_triangle(S, time_window)
    smooth_3 = smooth_gaussian(S, time_window)
    print(smooth_1)  # print the result of mean average
    print(strictly_increasing(smooth_1))
