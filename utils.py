import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def S(z, threshold):
    return numpy.sign(z) * numpy.maximum(0., numpy.abs(z) - threshold)


def loss(X, y, beta, alpha, lbda, groups):
    n, d = X.shape
    n_groups = numpy.max(groups) + 1
    l = numpy.linalg.norm(y - numpy.dot(X, beta)) ** 2 / (2 * n) + alpha * lbda * numpy.linalg.norm(beta, 1)
    for gr in range(n_groups):
        indices_group_k = groups == gr
        p_l = numpy.sqrt(numpy.sum(indices_group_k))
        l += (1 - alpha) * lbda * p_l * numpy.linalg.norm(beta[indices_group_k], 2)
    return l


def norm_non0(x):
    return numpy.maximum(numpy.linalg.norm(x), 1e-10)


def discard_group(X, y, beta, alpha, lbda, alpha_lambda, ind):
    n, d = X.shape
    X_k = X[:, ind]
    r_no_k = y - numpy.dot(X, beta) + numpy.dot(X_k, beta[ind])
    norm_2 = numpy.linalg.norm(S(numpy.dot(X_k.T, r_no_k) / n, alpha_lambda[ind]))
    p_l = numpy.sqrt(numpy.sum(ind))
    return norm_2 <= (1 - alpha) * lbda * p_l


if __name__ == "__main__":
    v = numpy.array([-1., 0.1, 2., 3, 4.5])
    alpha_lbda = 1.5
    print(S(v, alpha_lbda))
    print(S(0.1, 0))
    print(S(v, alpha_lbda * numpy.array([1, 1, 0, 0, 0])))