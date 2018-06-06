from sklearn.neighbors import BallTree
from random import sample
import numpy as np
from math import isnan

def hopkins(X):
    col = X.shape[1]
    print(col)

    row = X.shape[0]
    print(row)

    m = int(0.1 * row) # heuristic
    rand_X = sample(range(0, row, 1), m)
    print(rand_X)
    tree = BallTree(X[rand_X])
    # keep the same variance
    s = np.std(X)
    uni_sample = s * np.random.random_sample((m, col))
    uni_tree = BallTree(uni_sample)

    r_X = X[rand_X]

    ujd = []
    wjd = []

    for j in range(0, m):
       u_dist, _ = uni_tree.query([uni_sample[j]], k=2)
       print(u_dist[0])
       ujd.append(u_dist[0][1])
       w_dist, _ = tree.query([r_X[j]], k=2)
       print(w_dist[0])
       wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):
        print (ujd)
        print (wjd)
        H = 0

    return H