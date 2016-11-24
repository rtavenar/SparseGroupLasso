import numpy
import matplotlib.pyplot as plt
from utils import S, norm_non0, discard_group

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SGL:
    def __init__(self, groups, alpha, lbda, ind_sparse, max_iter_outer=10000, max_iter_inner=100, rtol=1e-6):
        self.ind_sparse = numpy.array(ind_sparse)
        self.groups = numpy.array(groups)
        self.alpha = alpha
        self.lbda = lbda
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rtol = rtol
        self.coef_ = None

    def fit(self, X, y):
        # Assumption: group ids are between 0 and max(groups)
        # Other assumption: ind_sparse is of dimension X.shape[1] and has 0 if the dimension should not be pushed
        # towards sparsity and 1 otherwise
        n_groups = numpy.max(self.groups) + 1
        n, d = X.shape
        assert d == self.ind_sparse.shape[0]
        alpha_lambda = self.alpha * self.lbda * self.ind_sparse
        self.coef_ = numpy.random.randn(d)
        t = 1. / (numpy.linalg.norm(X, 2) ** 2)  # Heuristic (?) from fabianp's code
        for iter_outer in range(self.max_iter_outer):
            beta_old = self.coef_.copy()
            for gr in range(n_groups):
                # 1- Should the group be zero-ed out?
                indices_group_k = self.groups == gr
                if discard_group(X, y, self.coef_, self.alpha, self.lbda, alpha_lambda, indices_group_k):
                    self.coef_[indices_group_k] = 0.
                else:
                    # 2- If the group is not zero-ed out, perform GD for the group
                    beta_k = self.coef_[indices_group_k]
                    p_l = numpy.sqrt(numpy.sum(indices_group_k))
                    for iter_inner in range(self.max_iter_inner):
                        grad_l = self._grad_l(X, y, indices_group_k)
                        tmp = S(beta_k - t * grad_l, t * alpha_lambda[indices_group_k])
                        tmp *= numpy.maximum(1. - t * (1 - self.alpha) * self.lbda * p_l / numpy.linalg.norm(tmp), 0.)
                        if numpy.linalg.norm(tmp - beta_k) / norm_non0(tmp) < self.rtol:
                            self.coef_[indices_group_k] = tmp
                            break
                        beta_k = self.coef_[indices_group_k] = tmp
            if numpy.linalg.norm(beta_old - self.coef_) / norm_non0(self.coef_) < self.rtol:
                break
        return self

    def _grad_l(self, X, y, indices_group):  # Linear Regression
        n, d = X.shape
        r = y - numpy.dot(X, self.coef_)
        return - numpy.dot(X[:, indices_group].T, r) / n

    def predict(self, X):
        return numpy.dot(X, self.coef_)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    @staticmethod
    def lambda_max(X, y, groups, alpha):
        # TODO: take ind_sparse into account in the computation + logistic regression variant
        n, d = X.shape
        n_groups = numpy.max(groups) + 1
        max_min_lambda = -numpy.inf
        for gr in range(n_groups):
            indices_group = groups == gr
            sqrt_p_l = numpy.sqrt(numpy.sum(indices_group))
            vec_A = numpy.abs(numpy.dot(X[:, indices_group].T, y)) / n
            if alpha > 0.:
                min_lambda = numpy.inf
                breakpoints_lambda = numpy.unique(vec_A / alpha)

                for l in breakpoints_lambda:
                    indices_nonzero = vec_A >= alpha * l
                    n_nonzero = numpy.sum(indices_nonzero)
                    a = n_nonzero * alpha ** 2 - (sqrt_p_l * (1. - alpha)) ** 2
                    b = - 2. * alpha * numpy.sum(vec_A[indices_nonzero])
                    c = numpy.sum(vec_A[indices_nonzero] ** 2)
                    delta = b ** 2 - 4 * a * c
                    if delta >= 0.:
                        candidate = (- b - numpy.sqrt(delta)) / (2 * a)
                        if candidate <= l:
                            min_lambda = candidate
                            break
            else:
                min_lambda = numpy.linalg.norm(numpy.dot(X[:, indices_group].T, y) / n) / sqrt_p_l
            if min_lambda > max_min_lambda:
                max_min_lambda = min_lambda
        return max_min_lambda

    @staticmethod
    def candidate_lambdas(X, y, groups, alpha, n_lambdas=5, lambda_min_ratio=.1):
        l_max = SGL.lambda_max(X, y, groups=groups, alpha=alpha)
        return numpy.logspace(numpy.log10(lambda_min_ratio * l_max), numpy.log10(l_max), num=n_lambdas)


class SGL_LogisticRegression(SGL):
    # Up to now, we assume that y is 0 or 1 (TODO: change that)
    def _grad_l(self, X, y, indices_group):
        n, d = X.shape
        p_y0 = SGL_LogisticRegression.__logistic(X, self.coef_)
        p_y1 = 1. - p_y0
        return numpy.sum(X[:, indices_group] * (y + p_y1).reshape((n, 1)), axis=0) / n


    @staticmethod
    def __logistic(X, beta):
        return 1. / (1. + numpy.exp(numpy.dot(X, beta)))


if __name__ == "__main__":
    n = 1000
    d = 20
    groups = numpy.array([0] * int(d / 2) + [1] * (d - int(d / 2)))
    alpha = .5
    epsilon = .001

    numpy.random.seed(0)
    X = numpy.random.randn(n, d)
    secret_beta = numpy.random.randn(d)
    ind_sparse = numpy.ones((d, ))
    for i in range(d):
        if groups[i] == 0:
            secret_beta[i] = 0

    y = numpy.dot(X, secret_beta)

    lambda_max = SGL.lambda_max(X, y, groups=groups, alpha=alpha)
    print(lambda_max)
    for l in [lambda_max - epsilon, lambda_max + epsilon]:
        model = SGL(groups=groups, alpha=alpha, lbda=l, ind_sparse=ind_sparse)
        model.fit(X, y)
        print(l, model.coef_)

    print(SGL.candidate_lambdas(X, y, groups=groups, alpha=alpha))