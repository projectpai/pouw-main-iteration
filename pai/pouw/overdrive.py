import mxnet.ndarray as nd


def calculate_overdrive(initial, final, tau):
    overdrive = 0
    for i, g in enumerate(initial):
        overdrive += (nd.abs(g - final[i]) > tau).sum().asscalar()
    return int(overdrive)


def clone_gradients(gradients):
    gradients_copy = []
    for g in gradients:
        gradients_copy.append(g.copy())

    return gradients_copy
