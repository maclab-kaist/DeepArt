

```python
import numpy, scipy, matplotlib.pyplot as plt, pandas, librosa
```

[&larr; Back to Index](index.html)

# NumPy and SciPy

The quartet of NumPy, SciPy, Matplotlib, and IPython is a popular combination in the Python world. We will use each of these libraries in this workshop.

## Tutorial

[NumPy](http://www.numpy.org) is one of the most popular libraries for numerical computing in the world. It is used in several disciplines including image processing, finance, bioinformatics, and more.  This entire workshop is based upon NumPy and its derivatives.

If you are new to NumPy, follow this [NumPy Tutorial](http://wiki.scipy.org/Tentative_NumPy_Tutorial).

[SciPy](http://docs.scipy.org/doc/scipy/reference/) is a Python library for scientific computing which builds on top of NumPy. If NumPy is like the Matlab core, then SciPy is like the Matlab toolboxes. It includes support for linear algebra, sparse matrices, spatial data structions, statistics, and more.

While there is a [SciPy Tutorial](http://docs.scipy.org/doc/scipy/reference/tutorial/index.html), it isn't critical that you follow it for this workshop.

## Special Arrays


```python
print numpy.arange(5)
```

    [0 1 2 3 4]



```python
print numpy.linspace(0, 5, 10, endpoint=False)
```

    [ 0.   0.5  1.   1.5  2.   2.5  3.   3.5  4.   4.5]



```python
print numpy.zeros(5)
```

    [ 0.  0.  0.  0.  0.]



```python
print numpy.ones(5)
```

    [ 1.  1.  1.  1.  1.]



```python
print numpy.ones((5,2))
```

    [[ 1.  1.]
     [ 1.  1.]
     [ 1.  1.]
     [ 1.  1.]
     [ 1.  1.]]



```python
print scipy.randn(5) # random Gaussian, zero-mean unit-variance
```

    [ -1.12009510e+00   2.15875646e-03  -7.93208376e-01  -1.02710782e+00
       2.37388108e+00]



```python
print scipy.randn(5,2)
```

    [[-0.60527349 -0.82200312]
     [-0.67330474 -0.12914043]
     [-0.71574719 -0.5962005 ]
     [-1.03690426  0.59078457]
     [-2.22983691 -1.70858604]]


## Slicing Arrays


```python
x = numpy.arange(10)
print x[2:4]
```

    [2 3]



```python
print x[-1]
```

    9


The optional third parameter indicates the increment value:


```python
print x[0:8:2]
```

    [0 2 4 6]



```python
print x[4:2:-1]
```

    [4 3]


If you omit the start index, the slice implicitly starts from zero:


```python
print x[:4]
```

    [0 1 2 3]



```python
print x[:999]
```

    [0 1 2 3 4 5 6 7 8 9]



```python
print x[::-1]
```

    [9 8 7 6 5 4 3 2 1 0]


## Array Arithmetic


```python
x = numpy.arange(5)
y = numpy.ones(5)
print x+2*y
```

    [ 2.  3.  4.  5.  6.]


`dot` computes the dot product, or inner product, between arrays or matrices.


```python
x = scipy.randn(5)
y = numpy.ones(5)
print numpy.dot(x, y)
```

    -4.36027379404



```python
x = scipy.randn(5,3)
y = numpy.ones((3,2))
print numpy.dot(x, y)
```

    [[ 0.9351335   0.9351335 ]
     [-4.22851009 -4.22851009]
     [-2.66983557 -2.66983557]
     [ 3.18545804  3.18545804]
     [ 1.82532797  1.82532797]]


## Boolean Operations


```python
x = numpy.arange(10)
print x < 5
```

    [ True  True  True  True  True False False False False False]



```python
y = numpy.ones(10)
print x < y
```

    [ True False False False False False False False False False]


## Distance Metrics


```python
from scipy.spatial import distance
print distance.euclidean([0, 0], [3, 4])
print distance.sqeuclidean([0, 0], [3, 4])
print distance.cityblock([0, 0], [3, 4])
print distance.chebyshev([0, 0], [3, 4])
```

    5.0
    25.0
    7
    4


The cosine distance measures the angle between two vectors:


```python
print distance.cosine([67, 0], [89, 0])
print distance.cosine([67, 0], [0, 89])
```

    0.0
    1.0


## Sorting

NumPy arrays have a method, `sort`, which sorts the array *in-place*.


```python
x = scipy.randn(5)
print x
x.sort()
print x
```

    [ 0.70589021  0.14767722  0.06884379  0.37189002  0.43313129]
    [ 0.06884379  0.14767722  0.37189002  0.43313129  0.70589021]


`numpy.argsort` returns an array of indices, `ind`, such that `x[ind]` is a sorted version of `x`.


```python
x = scipy.randn(5)
print x
ind = numpy.argsort(x)
print ind
print x[ind]
```

    [ 0.9443719   0.2831604   0.85627     0.22827583 -0.03939166]
    [4 3 1 2 0]
    [-0.03939166  0.22827583  0.2831604   0.85627     0.9443719 ]


[&larr; Back to Index](index.html)
