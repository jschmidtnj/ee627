#!/usr/bin/env python
"""
matrix factorization code
"""

import numpy as np


def matrix_factorization(R, P, Q, K, steps, alpha, beta):
    """
    matrix factorization
    """
    Q_t = Q.T
    for step in range(steps):
        for i, elem_i in enumerate(R):
            for j, elem_j in enumerate(elem_i):
                if elem_j <= 0:
                    continue
                new_i_j = elem_j - np.dot(P[i, :], Q_t[:, j])
                for k in range(K):
                    P[i][k] = P[i][k] + alpha * \
                        (2 * new_i_j * Q_t[k][j] - beta * P[i][k])
                    Q_t[k][j] = Q_t[k][j] + alpha * \
                        (2 * new_i_j * P[i][k] - beta * Q_t[k][j])
        err = 0.
        for i, elem_i in enumerate(R):
            for j, elem_j in enumerate(elem_i):
                if elem_j <= 0:
                    continue
                err = err + (elem_j - np.dot(P[i, :], Q_t[:, j])) ** 2
                for k in range(K):
                    err = err + (beta / 2) * (P[i][k] ** 2 + Q_t[k][j] ** 2)
        if err < 1e-4:
            break

    Q = Q_t.T
    return P, Q


def test():
    """
    test script
    """
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    nRow, nCol = len(R), len(R[0])
    K = 2
    P = np.random.rand(nRow, K)
    Q = np.random.rand(nCol, K)

    steps = 5000
    alpha = 0.0002
    beta = 0.02

    nP, nQ = matrix_factorization(R, P, Q, K, steps, alpha, beta)
    print(np.dot(nP, nQ.T))
    print(R)

    steps = 10000
    alpha = 0.0002
    beta = 0.02

    nP, nQ = matrix_factorization(R, P, Q, K, steps, alpha, beta)
    print(np.dot(nP, nQ.T))
    print(R)

if __name__ == '__main__':
    test()
