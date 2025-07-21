import numpy as np
import sys

def dynamic_time_warping(x: np.ndarray, y: np.ndarray, w: float = -1):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    if len(x) > len(y):
        x, y = y, x

    lx, ly = len(x), len(y)
    r, c = lx + 1, ly + 1

    w = max(1, int(w * max(lx, ly))) if w >= 0 else max(lx, ly)

    D = np.zeros((r, c), dtype=np.float64)
    D[0, 1:] = sys.float_info.max
    D[1:, 0] = sys.float_info.max

    dist = np.square(x[:, np.newaxis] - y).sum(axis=2)
    D[1:, 1:] = dist

    for i in range(1, r):
        j_start = max(1, i - w)
        j_stop = min(c, i + w + 1)
        if i - w - 1 >= 0:
            D[i, i - w - 1] = sys.float_info.max

        for j in range(j_start, j_stop):
            D[i, j] += min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        if j_stop < c:
            D[i, j_stop] = sys.float_info.max

    return np.sqrt(D[lx, ly]), D
