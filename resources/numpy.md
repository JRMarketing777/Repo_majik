Here's a comprehensive cheatsheet for NumPy in markdown format:

# NumPy Cheatsheet

## Installation and Import

```python
pip install numpy
import numpy as np
```

## Array Creation

```python
# From list
a = np.array([1, 2, 3])

# With a range
b = np.arange(10)  # [0, 1, 2, ..., 9]

# With specific step
c = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Zeros and ones
d = np.zeros((3, 3))
e = np.ones((2, 2))

# Identity matrix
f = np.eye(3)

# Random numbers
g = np.random.rand(3, 3)  # Uniform distribution [0, 1)
h = np.random.randn(3, 3)  # Normal distribution

# Linspace
i = np.linspace(0, 1, 5)  # 5 evenly spaced numbers from 0 to 1
```

## Array Attributes and Methods

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape  # Dimensions of the array
a.dtype  # Data type of elements
a.size   # Total number of elements
a.ndim   # Number of dimensions

a.reshape(3, 2)  # Reshape array
a.flatten()      # Flatten to 1D array
a.T              # Transpose
```

## Array Indexing and Slicing

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

a[0, 1]     # Element at row 0, column 1
a[0]        # First row
a[:, 1]     # Second column
a[0:2, 1:3] # Subarray

# Boolean indexing
mask = a > 5
a[mask]  # All elements greater than 5
```

## Array Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
a + b
a - b
a * b
a / b

# Scalar operations
a + 2
a * 2

# Matrix multiplication
np.dot(a, b)
a @ b  # Python 3.5+

# Universal functions
np.sqrt(a)
np.exp(a)
np.sin(a)
```

## Broadcasting

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])

a + b  # b is broadcast to match a's shape
```

## Array Manipulation

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

np.concatenate((a, b), axis=0)  # Vertical concatenation
np.concatenate((a, b.T), axis=1)  # Horizontal concatenation

np.split(a, 2, axis=0)  # Split into two along first axis
np.hsplit(a, 2)         # Horizontal split
np.vsplit(a, 2)         # Vertical split
```

## Statistical Functions

```python
a = np.array([[1, 2], [3, 4]])

np.mean(a)        # Mean of all elements
np.mean(a, axis=0)  # Mean of each column
np.std(a)         # Standard deviation
np.var(a)         # Variance
np.min(a)         # Minimum
np.max(a)         # Maximum
np.argmin(a)      # Index of minimum
np.argmax(a)      # Index of maximum
```

## Linear Algebra

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.linalg.inv(a)    # Inverse
np.linalg.det(a)    # Determinant
np.linalg.eig(a)    # Eigenvalues and eigenvectors
np.linalg.solve(a, b)  # Solve linear system Ax = b
```

## File I/O

```python
# Save array to file
np.save('array.npy', a)

# Load array from file
b = np.load('array.npy')

# Save multiple arrays
np.savez('arrays.npz', a=a, b=b)

# Load multiple arrays
data = np.load('arrays.npz')
a = data['a']
b = data['b']
```

## Advanced Indexing

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Integer array indexing
indices = np.array([[0, 1], [1, 2]])
a[indices]  # Returns array([2, 6])

# Boolean array indexing
mask = a > 5
a[mask]  # Returns array([6, 7, 8, 9])
```

This cheatsheet covers the most commonly used NumPy operations and functions. Remember to refer to the official NumPy documentation for more detailed information and advanced usage.
