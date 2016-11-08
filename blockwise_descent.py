import numpy
import blockwise_descent_semisparse

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SGL(blockwise_descent_semisparse.SGL):
    def __init__(self, groups, alpha, lbda, max_iter_outer=10000, max_iter_inner=100, rtol=1e-6):
        self.ind_sparse = numpy.ones((len(groups), ))
        self.groups = groups
        self.alpha = alpha
        self.lbda = lbda
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rtol = rtol
        self.coef_ = None
