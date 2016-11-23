import numpy as np
import pylab as plt

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)

alpha = 0.5
Z_lasso = np.abs(X) + np.abs(Y)
Z_ridge = X ** 2 + Y ** 2
Z_grouplasso = (1 - alpha) * np.sqrt(X ** 2 + Y ** 2) + alpha * Z_lasso
Z_grouplasso_semisparse = (1 - alpha) * np.sqrt(X ** 2 + Y ** 2) + np.abs(X)

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.contour(X, Y, Z_lasso)
plt.title("Lasso")

plt.subplot(1, 4, 2)
plt.contour(X, Y, Z_ridge)
plt.title("Ridge")

plt.subplot(1, 4, 3)
plt.contour(X, Y, Z_grouplasso)
plt.title("Sparse-group lasso")

plt.subplot(1, 4, 4)
plt.contour(X, Y, Z_grouplasso_semisparse)
plt.title("Semi-sparse-group lasso (no sparsity on Y)")

plt.savefig("penalties.pdf")