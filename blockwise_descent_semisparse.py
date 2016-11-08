import numpy
from utils import S, norm_non0

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SGL:
    def __init__(self, groups, alpha, lbda, ind_sparse, max_iter_outer=10000, max_iter_inner=100, rtol=1e-6):
        self.ind_sparse = ind_sparse
        self.groups = groups
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
        alpha_lambda = self.alpha * self.lbda * self.ind_sparse
        self.coef_ = numpy.random.randn(d)
        t = 1. / (numpy.linalg.norm(X, 2) ** 2)  # Heuristic (?) from fabianp's code
        for iter_outer in range(self.max_iter_outer):
            beta_old = self.coef_.copy()
            # print(iter_outer, numpy.linalg.norm(y - numpy.dot(X, beta)))
            for gr in range(n_groups):
                # 1- Should the group be zero-ed out?
                indices_group_k = self.groups == gr
                X_k = X[:, indices_group_k]
                r_no_k = y - numpy.dot(X, self.coef_) + numpy.dot(X_k, self.coef_[indices_group_k])
                norm_2 = numpy.linalg.norm(S(numpy.dot(X_k.T, r_no_k) / n, alpha_lambda[indices_group_k]))
                p_l = numpy.sqrt(numpy.sum(indices_group_k))
                if norm_2 <= (1 - self.alpha) * self.lbda * p_l:
                    self.coef_[indices_group_k] = 0.
                else:
                    # 2- If the group is not zero-ed out, perform GD for the group
                    beta_k = self.coef_[indices_group_k]
                    for iter_inner in range(self.max_iter_inner):
                        r = y - numpy.dot(X, self.coef_)
                        grad_l = - numpy.dot(X_k.T, r) / n  # To be changed if logistic regression
                        tmp = S(beta_k - t * grad_l, t * alpha_lambda[indices_group_k])
                        tmp *= numpy.maximum(1. - t * (1 - self.alpha) * self.lbda * p_l / numpy.linalg.norm(tmp), 0.)
                        if numpy.linalg.norm(tmp - beta_k) / norm_non0(tmp) < self.rtol:
                            self.coef_[indices_group_k] = tmp
                            break
                        beta_k = self.coef_[indices_group_k] = tmp
            if numpy.linalg.norm(beta_old - self.coef_) / norm_non0(self.coef_) < self.rtol:
                break
        return self