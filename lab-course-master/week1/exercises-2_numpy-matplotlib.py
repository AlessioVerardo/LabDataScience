# %% [markdown]
# ## NumPy arrays & Matplotlib
#
# Operations and indexing are not vectorized on python's native structures.
# Numpy brings an ndarray (N-dimensional array) structure which implements vectorization.
#
# In practice, this means you can do things like this:

# %%
import numpy as np
arr = np.array([10, 5, 12, 30])
arr * 2 + 1

# %% [markdown]
# ### Exercise 1.1: Fun with numpy.linspace()

# %% [markdown]
# Create an array of of 20 elements with values ranging from 0 to 1, excluding 0 and 1

# %%
# Write your code here
x = ...
print(x)

# %% [markdown]
# ### Exercise 1.2: Fun with numpy.reshape()

# %% [markdown]
# Form the following 2-D array (without typing it in explicitly):
#
# $$
# \begin{pmatrix}
#    1 & 6 & 11 \\
#    2 & 7 & 12 \\
#    3 & 8 & 13 \\
#    4 & 9 & 14 \\
#    5 & 10 & 15 
# \end{pmatrix}
# $$
#
# and generate a new array containing its 2nd and 4th rows.

# %% [markdown]
# (Copyright http://www.scipy-lectures.org/intro/numpy/exercises.html)

# %%
# Write your code here
x1 = ...
x2 = ...
print(x2)

# %% [markdown]
# ### Exercise 1.3: Harder one!

# %% [markdown]
# Generate a 10 x 3 array of random numbers in [0,1].
#
# For each row, pick the number closest to 0.5.
#
#     - Use abs and argmin to find the column j closest for each row

# %% [markdown]
# (Copyright http://www.scipy-lectures.org/intro/numpy/exercises.html)

# %%
# Write your code here
x1 = ...
print(x1)
c05 = ...
print(c05)

# %% [markdown]
# ### Exercise 1.4: Sparse Matrices

# %% [markdown]
# Create a random array (element values between -1 and 1) with 10,000 elements, and two random arrays with 9,999 elements. Create a scipy sparse matrix (see [here](https://docs.scipy.org/doc/scipy/reference/sparse.html) for different types of sparse matrices) that has the first array on its main diagonal, the second one next to the main diagonal and above, and the third one just below the main diagonal. There are multiple ways to do this:
# * Using the [diags](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html) function
# * Using the constructors for sparse matrices (see [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix) for an example)
#
# This sparse matrix will take up much less space than a dense matrix would. To see this, you can densify this matrix (if you have enough RAM, be careful because it will take about 1 GB) and compare its size to the original sparse matrix (see [here](https://stackoverflow.com/questions/11784329/python-memory-usage-of-numpy-arrays) for details on getting the size in bytes).
#
# Now, perform Singular Value Decomposition (SVD) on this matrix using one of SciPy's built-in functions. SVD is conceptually very similar to Principal Component Analysis (PCA) that we discussed during today's lecture (and is actually one way of calculating a PCA). Look at the singular values and how quickly they decline, and determine the number of dimensions you would keep to have a good approximation of the original matrix.

# %%

# Write your code here

# %% [markdown]
# ### Exercise 2: One-hot encoding

# %% [markdown]
# There are many applications to the classification and analysis of DNA sequences, but DNA (and strings in general) cannot be directly fed into numerical models. We will use the python scientific stack to prepare them in a format fit for analysis.
#
# We need to represent the set of nucleotides $n = \{A,C,G,T\}$ as numeric values.
# The simplest method would be to map each nucleotide to a number, e.g. $\{A,C,G,T\} \rightarrow \{0,1,2,3\}$.
#
# However, this encoding is inappropriate, as it implies an ordinal relationship between nucleotides.
#
# In this case, it is more appropriate to use one-hot encoding. In this encoding, we convert each nucleotide to an array of 1 and 0s.
#
# Although this transformation increases the number of dimensions, it ensures all nucleotides are equidistant to each other.

# %% [markdown]
# Sequences are provided below as a 2D numpy array of shape (n_sequences x len_sequences), all sequences are of the same length.
#
# Convert DNA sequences into one-hot encoded features. Make use of numpy and vectorized operations and indexing. Avoid explicit loops to ensure good performances.
#
# %%
from typing import Iterable, Iterator, List, Dict
import itertools as it
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# Loading example sequences
seqs_arr = np.loadtxt('../data/promoter-sequences-classification/promoter_sequences.csv', delimiter=',', dtype=str)
seqs_arr = np.apply_along_axis(lambda x: [[c for c in s.strip()] for s in x], 0, seqs_arr[:, 2])
seqs_arr

# %%
import numpy as np

# Write your code here
def encode_onehot(seq: 'np.ndarray[str]') -> 'np.ndarray[int]':
    """One hot encode a 2D array of strings into a 3D array of ints.
    
    >>> x = encode_onehot(np.array([['a', 'c'], ['t', 't']]))
    >>> encode_onehot(x)
    array([[[1., 0., 0., 0.],
        [0., 1., 0., 0.]],

       [[0., 0., 0., 1.],
        [0., 0., 0., 1.]]])
    """
    # Write your code here
    # Tip: get a mapping between nucleotides and numeric arrays
    # np.eye() can be useful
    ...



# %%
# Test your function
oh_seqs = encode_onehot(seqs_arr)
oh_seqs


# %% [markdown]
# Use matplotlib's `imshow` function to visualize the average onehot encoding profile of your sequences.
# This can be plotted as a heatmap where rows are the nucleotides (A,C,G,T), columns are sequence positions and pixel values are the average abundance of a nucleotide at a position.
#
# Do you notice any interesting pattern ?

# %%
# Write your code here
...

# %% [markdown]
# ## Bonus:
#
# The first 53 sequences are bacterial promoter sequences (i.e. regulatory sequences located directly before the start of a gene), whereas the rest are random bacterial sequences.
#
# Try plotting either group to see the difference.
#
# > Note: All promoter sequences contain positions -50 to +7 relative the gene start.

# %%
# Write your code here
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(15, 6))
...
