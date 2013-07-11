aim3-lr
=======

AIM3 project - Logistic Regression

Contains two maven modules:

1) aim3-logreg: Different (parallel) training methods for Logistic Regression
- Hadoop job for Ensemble + SGD (mahout)
- Hadoop job for Batch Gradient descent
- Hadoop job for L-BFGS (uses Parallel Gradient-computation job)

2) aim3-logreg-rcv1: Flexible preprocessing for rcv1-v2 (base on cuttlefish)
- contains v2 patch
- flexible time based splits
- multiple tf-idf weighting options
- ...
