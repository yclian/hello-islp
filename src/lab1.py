import numpy as np
import random

from faker import Faker
fake = Faker()

# -----------------------------------------------------------------------------
## array operations
# -----------------------------------------------------------------------------

x = np.array(range(1, 7))
print('x:\n', x)

x_reshape = x.reshape(3, 2)
print('x_reshape:\n', x_reshape)

x_transpose = x_reshape.transpose()
print('x_transpose:\n', x_transpose)

x_t_multiply = x_transpose * x.reshape(2, 3)
print('x_transpose * 2x3:\n', x_t_multiply)

print(x_t_multiply[0, 1])
print(x_t_multiply[1, 2])

print('x squared:\n', x**2)

# -----------------------------------------------------------------------------
## random numbers
# -----------------------------------------------------------------------------

x = np.random.normal(size = 10)
print('x:\n', x)
print('x_mean:\n', np.mean(x))
print('x_std:\n', np.std(x))

y = x + np.random.normal(loc = 10, scale = 1, size = 10)
print('y:\n', y)
print('y_mean:\n', np.mean(y))
print('y_std:\n', np.std(y))
print('corr:\n', np.corrcoef(x, y))

# -----------------------------------------------------------------------------
## standard normal distribution
# -----------------------------------------------------------------------------

rng1 = np.random.default_rng(1337)
rng2 = np.random.default_rng(1337)
rng3 = np.random.default_rng(7331)

print('rng1\'s normal:\n', rng1.normal(scale = 1, size = 2))
print('rng1\'s normal:\n', rng1.normal(scale = 1, size = 2))
print('rng2\'s normal:\n', rng2.normal(scale = 1, size = 2))
print('rng2\'s normal:\n', rng2.normal(scale = 1, size = 2))
print('rng3\'s normal:\n', rng3.normal(scale = 1, size = 2))

# y is a sample of 10 random numbers generated from the standard normal 
#   distribution using the rng1.standard_normal(10) function. 
# y is a sample, not a population.
y = rng1.standard_normal(10)

print('y:\n', y)
print('y mean:\n', np.mean(y))
print('y mean:\n', y.mean())
print('y variance:\n', np.var(y))
print('y variance:\n', y.var())
# The ddof argument stands for "degrees of freedom."
# The degrees of freedom is the number of independent observations in a sample.
# To calculate the unbiased estimator, we set the ddof argument to 1. 
# This tells np.var() to divide the sum of the squared deviations from the mean 
#   by the sample size minus 1.
print('y variance (unbiased):\n', np.var(y, ddof=1)) 

print('y\'s deviations from the mean:\n', (y - y.mean()))
print('y\'s squared deviations from the mean:\n', (y - y.mean()) ** 2)
print('y\'s sum of squared deviations from the mean:\n', np.sum((y - y.mean()) ** 2))
# The np.mean() function calculates the arithmetic mean of the elements in an 
# array. It does not take into account the sample size when calculating the 
# mean.
print('y variance:\n', np.mean((y - y.mean()) ** 2))
# When calculating the sample variance, we should use len(y) - 1 as the divisor 
# instead of len(y). This is because the sample variance is an unbiased 
# estimator of the population variance, meaning that it will, on average, 
# estimate the true population variance correctly.
print('unbiased estimator of y\'s variance:\n', np.sum((y - y.mean()) ** 2)/(len(y) - 1))
print('biased estimator of y\'s variance:\n', np.sum((y - y.mean()) ** 2)/len(y))

# the square root of the variance is the standard deviation. This relationship 
# holds true for both population and sample variances.
print('', np.sqrt(np.var(y)))
print('', np.std(y))

# 2d array of 10x2 random numbers from the standard normal distribution
X = rng1.standard_normal((3, 2))
print('X as 10x2 array:\n', X)
# this returns a single number for a 2D array is because it calculates the 
# standard deviation across all elements in the array, regardless of their 
# dimensions. In other words, it treats the 2D array as a single, flattened 
# array of numbers.
print('standard deviation of X:\n', np.std(X))
print('mean of X (flattened):\n', X.mean())
print('mean of X:\n', X.mean(0))
print('mean of X:\n', X.mean(axis=0))