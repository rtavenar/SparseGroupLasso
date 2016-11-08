# Sparse-Group Lasso

This code is my implementation (in Python) of the methods presented in the paper:
> A Sparse-group Lasso.
> Noah Simon, Jerome Friedman, Trevor Hastie, Rob Tibshirani
> <http://www.stanford.edu/~hastie/Papers/SGLpaper.pdf>

`*_semisparse` variants of the methods correspond to cases where one would allow sparsity for some dimensions but not all (for the L1-norm penalty).
In these cases, an indicator vector `ind_sparse` should be given that has `0` values for dimensions that should not be pushed towards sparsity and `1` values otherwise.
Typical function calls are given in the `test_sgl.py` script (models are `sklearn`-like objects).

## Yet to do
* Add proper docstrings