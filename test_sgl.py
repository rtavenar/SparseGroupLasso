import numpy
import subgradients, subgradients_semisparse, blockwise_descent, blockwise_descent_semisparse

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

n = 1000
d = 20
groups = numpy.array([0] * int(d / 2) + [1] * (d - int(d / 2)))
alpha = 0.
lbda = 0.1

X = numpy.random.randn(n, d)
secret_beta = numpy.random.randn(d)
ind_sparse = numpy.zeros((d, ))
for i in range(d):
    if groups[i] == 0 or i % 2 == 0:
        secret_beta[i] = 0
    if i % 2 != 0:
        ind_sparse[i] = 1

y = numpy.dot(X, secret_beta)

#model = subgradients.SGL(groups=groups, alpha=0., lbda=0.1)
#model = subgradients_semisparse.SGL(groups=groups, alpha=0.1, lbda=0.1, ind_sparse=ind_sparse)
#model = blockwise_descent.SGL(groups=groups, alpha=0., lbda=0.1)
model = blockwise_descent_semisparse.SGL(groups=groups, alpha=alpha, lbda=lbda, ind_sparse=ind_sparse)

model.fit(X, y)
beta_hat = model.coef_

print("Norm of the error on beta: %f" % numpy.linalg.norm(secret_beta - beta_hat))
print(beta_hat)
print(secret_beta)

epsilon = .001
lambda_max = blockwise_descent_semisparse.SGL.lambda_max(X, y, groups=groups, alpha=alpha, ind_sparse=ind_sparse)
print("lambda_max = %f" % lambda_max)
for l in [lambda_max - epsilon, lambda_max + epsilon]:
    model = blockwise_descent_semisparse.SGL(groups=groups, alpha=alpha, lbda=l, ind_sparse=ind_sparse)
    model.fit(X, y)
    print("lambda = %f, beta=%s" % (l, str(model.coef_)))

print("Path of lambdas:", blockwise_descent_semisparse.SGL.candidate_lambdas(X, y, groups=groups, alpha=alpha))
