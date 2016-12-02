import numpy
import matplotlib.pyplot as plt
from blockwise_descent_semisparse import SGL_LogisticRegression
from utils import S

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

n = 100
d = 2
xylim = 10

numpy.random.seed(0)
X = numpy.random.randn(n, d)
secret_beta = numpy.random.randn(d)
groups = numpy.zeros((d, ))

ind_sparse = numpy.ones((d, ))

y = numpy.ones((n, ))
y[numpy.exp(numpy.dot(X, secret_beta)) < 1.] = 0.

beta_path = numpy.empty((100, 2))
beta_path[0] = [0, 8]
t = n / (numpy.linalg.norm(X, 2) ** 2)
alpha = 0.

beta0, beta1 = numpy.meshgrid(numpy.linspace(-xylim, xylim, 100), numpy.linspace(-xylim, xylim, 100))
beta_full = numpy.c_[beta0.ravel(), beta1.ravel()]

plt.figure(figsize=(15, 5))
for idx, lbda in enumerate([0., .1, 1.]):
    for i in range(1, beta_path.shape[0]):
        model = SGL_LogisticRegression(groups=[0, 0], alpha=alpha, lbda=lbda, ind_sparse=ind_sparse)
        model.coef_ = beta_path[i - 1]
        grad_l = model._grad_l(X, y, groups == 0)
        tmp = S(model.coef_ - t * grad_l, t * alpha * lbda * numpy.ones((d, )))
        tmp *= numpy.maximum(1. - t * (1. - alpha) * lbda / numpy.linalg.norm(tmp), 0.)
        beta_path[i] = tmp
    model = SGL_LogisticRegression(groups=[0, 0], alpha=alpha, lbda=lbda, ind_sparse=ind_sparse)
    loss = numpy.empty(beta_full[:, 0].shape)
    for i, beta in enumerate(beta_full):
        model.coef_ = beta
        loss[i] = model.loss(X, y)

    plt.subplot(1, 3, idx + 1)
    plt.plot(beta_path[:, 0], beta_path[:, 1], "w-x")
    plt.axvline(x=0., color="k", linestyle="dashed")
    plt.axhline(y=0., color="k", linestyle="dashed")
    plt.contourf(beta0, beta1, loss.reshape(beta0.shape))
    plt.scatter(secret_beta[0], secret_beta[1], color="r")
    plt.xlim(-xylim, xylim)
    plt.ylim(-xylim, xylim)
    plt.title("lambda=%.1f" % lbda)
plt.savefig("logistic_cost.pdf")
