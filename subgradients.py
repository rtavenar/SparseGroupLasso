import numpy
import subgradients_semisparse

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SGL(subgradients_semisparse.SGL):
    def __init__(self, groups, alpha, lbda, max_iter=1000, rtol=1e-6):
        self.ind_sparse = numpy.ones((len(groups), ))
        self.groups = numpy.array(groups)
        self.alpha = alpha
        self.lbda = lbda
        self.max_iter = max_iter
        self.rtol = rtol
        self.coef_ = None
