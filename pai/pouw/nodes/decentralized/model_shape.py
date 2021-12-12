def get_shape(model):
    return [
        [w.shape.dims[0].value] if w.shape.ndims == 1 else [w.shape.dims[0].value, w.shape.dims[1].value] for w in
        model.trainable_weights]


def get_ranges(structure):
    ranges = []
    cumulated = 0
    for itm in structure:
        prev = cumulated
        cumulated += (itm[0] if len(itm) == 1 else itm[0] * itm[1])
        ranges.append((prev, cumulated - 1))
    return ranges
