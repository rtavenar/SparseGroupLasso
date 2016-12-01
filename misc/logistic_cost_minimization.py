import numpy
import matplotlib.pyplot as plt
from blockwise_descent_semisparse import SGL_LogisticRegression

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

n = 1000
d = 2
xylim = 10

numpy.random.seed(0)
X = numpy.random.randn(n, d)
secret_beta = numpy.random.randn(d)

y = numpy.ones((n, ))
y[numpy.exp(numpy.dot(X, secret_beta)) < 1.] = 0.

beta_path = numpy.empty((100, 2))
beta_path[0] = [0, 8]
for i in range(1, beta_path.shape[0]):
    model = SGL_LogisticRegression(groups=0, alpha=0, lbda=0, ind_sparse=0)
    model.coef_ = beta_path[i - 1]
    beta_path[i] = beta_path[i - 1] - 2. * model._grad_l(X, y, numpy.ones((d, )) > 0.)
    #print(numpy.sum(y == model.predict(X)))

beta0, beta1 = numpy.meshgrid(numpy.linspace(-xylim, xylim, 100), numpy.linspace(-xylim, xylim, 100))
beta_full = numpy.c_[beta0.ravel(), beta1.ravel()]

model = SGL_LogisticRegression(groups=0, alpha=0, lbda=0, ind_sparse=0)
model.coef_ = beta_full
loss = model._unregularized_loss(X, y)

plt.contourf(beta0, beta1, loss.reshape(beta0.shape))
plt.scatter(secret_beta[0], secret_beta[1], color="r")
plt.plot(beta_path[:, 0], beta_path[:, 1], "w-x")
plt.xlim(-xylim, xylim)
plt.ylim(-xylim, xylim)
plt.savefig("logistic_cost.pdf")
