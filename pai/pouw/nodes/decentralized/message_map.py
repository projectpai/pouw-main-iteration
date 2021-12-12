import tensorflow as tf


def is_upper_threshold(var):
    return var & (1 << 31) > 0


def decode_map_local(map_local, structure, ranges):
    upper_coord = [[] for i in range(len(ranges))]
    lower_coord = [[] for i in range(len(ranges))]
    for m in map_local:
        is_upper = is_upper_threshold(m)
        if is_upper:
            m &= (~(1 << 31))

        for i, r in enumerate(ranges):
            if r[0] <= m <= r[1]:
                m -= r[0]
                if len(structure[i]) == 1:
                    if is_upper:
                        upper_coord[i].append([m])
                    else:
                        lower_coord[i].append([m])
                else:
                    if is_upper:
                        upper_coord[i].append([m // structure[i][1], m % structure[i][1]])
                    else:
                        lower_coord[i].append([m // structure[i][1], m % structure[i][1]])
                break

    return upper_coord, lower_coord


def rebuild_delta_local(map_local, trainable_weights, tau, structure, ranges):
    delta_local = []
    coordinates = decode_map_local(map_local, structure, ranges)
    for idx, up_indices in enumerate(coordinates[0]):
        up_updates = tf.fill([len(up_indices)], tau)
        dw_updates = tf.fill([len(coordinates[1][idx])], -tau)
        updates = tf.concat([up_updates, dw_updates], -1)

        dw_indices = coordinates[1][idx]
        indices = up_indices + dw_indices

        if len(indices) > 0:
            delta_local_row = tf.scatter_nd(indices, updates, trainable_weights[idx].shape)
        else:
            delta_local_row = tf.zeros_like(trainable_weights[idx])
        delta_local.append(delta_local_row)
    return delta_local
