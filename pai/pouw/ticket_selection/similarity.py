from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim
