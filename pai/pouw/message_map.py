import numpy as np
from mxnet import ndarray as nd


def calculate_indexes(current_index, shape, gradients_cumulative, coordinates):
    if len(coordinates) == 1:
        return gradients_cumulative[current_index] + coordinates[0]
    if len(coordinates) == 2:
        return gradients_cumulative[current_index] + (coordinates[0] * shape[1] + coordinates[1])
    return None


def build_message_map(idx, gradients_cumulative, numpy_delta_positive, numpy_delta_negative,
                      zero_setter, one_setter):
    map_ups = np.where(numpy_delta_positive == 1)
    ups = calculate_indexes(idx, numpy_delta_positive.shape, gradients_cumulative, map_ups)
    ups = one_setter(ups)

    map_downs = np.where(numpy_delta_negative == 1)
    downs = calculate_indexes(idx, numpy_delta_negative.shape, gradients_cumulative, map_downs)
    downs = zero_setter(downs)

    return np.concatenate((ups, downs))


def get_indices(ups_or_downs, gradients_cumulative, gradients_blueprint):
    first_indices = [
        0 if t == 0 else len(gradients_cumulative) - 1 if gradients_cumulative[-1] <= t else \
            list(map(lambda x: x > t, gradients_cumulative)).index(True) - 1 for t in ups_or_downs]

    offsets = [(t - gradients_cumulative[first_indices[i]]) for i, t in enumerate(ups_or_downs)]

    all_indices = []
    for idx, t in enumerate(ups_or_downs):
        if len(gradients_blueprint[first_indices[idx]]) == 1:
            all_indices.append((first_indices[idx], offsets[idx],))
        elif len(gradients_blueprint[first_indices[idx]]) == 2:
            all_indices.append(
                (first_indices[idx], offsets[idx] // gradients_blueprint[first_indices[idx]][1],
                 offsets[idx] % gradients_blueprint[first_indices[idx]][1]))

    return all_indices


def decode_message_map(ctx, weight_indices, gradients_blueprint, gradients_cumulative, tau, zero_setter):
    message_map = np.array(weight_indices)
    sign_detector = np.vectorize(lambda int_type: int_type & (1 << 31) > 0, otypes=[np.bool])
    signs = sign_detector(message_map)
    ups = zero_setter(message_map[signs])
    downs = message_map[np.logical_not(signs)]

    peer_gradients = [nd.zeros(shape=blueprint, ctx=ctx) for blueprint in gradients_blueprint]
    up_indices = get_indices(ups, gradients_cumulative, gradients_blueprint)
    down_indices = get_indices(downs, gradients_cumulative, gradients_blueprint)

    for up in up_indices:
        if len(up) == 2:
            peer_gradients[up[0]][up[1]] = tau
        elif len(up) == 3:
            peer_gradients[up[0]][up[1]][up[2]] = tau

    for down in down_indices:
        if len(down) == 2:
            peer_gradients[down[0]][down[1]] = -tau
        elif len(down) == 3:
            peer_gradients[down[0]][down[1]][down[2]] = -tau

    return peer_gradients