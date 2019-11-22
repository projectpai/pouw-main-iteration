def count_ups(tau, layers):
    total = 0
    for layer in layers:
        total += ((layer == tau).sum().asscalar())

    return total


def count_downs(tau, layers):
    return count_ups(-tau, layers)
