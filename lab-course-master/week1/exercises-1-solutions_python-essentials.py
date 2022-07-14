# %% [markdown]
# # Crash course in Python

# %% [markdown]
# ## Generators
#
# Generator functions are a special kind of function that return a lazy iterator. They were first introduced into Python in [PEP 255](https://www.python.org/dev/peps/pep-0255) which states the motivation is to
# > provide a kind of function that can return an intermediate result ("the next value") to its caller, but maintaining the function's local state so that the function can be resumed again right where it left off.
#
# We won't explore every syntax detail here but give you a general idea of what generators are and how to construct and use them. Let's have look at a simple example from [PEP 255](https://www.python.org/dev/peps/pep-0255). The following example creates an infinite generator for Fibonacci sequence:
# ```python
# def fib():
#     a, b = 0, 1
#     while 1:
#         yield b
#         a, b = b, a+b
# ```
# You can call `next()` on the generator object directly:
# ```python
# gen = fib()  
# next(gen)    # initialize the state: a = 0, b = 1, yield b
# # 1
# next(gen)    # update the state: a = 1, b = 0 + 1 = 1, yield b
# # 1
# next(gen)    # update the state: a = 1, b = 1 + 1 = 2, yield b
# # 2
# next(gen)    # update the state: a = 2, b = 1 + 2 = 3, yield b
# # 3
# next(gen)    # update the state: a = 3, b = 2 + 3 = 5, yield b
# # 5
# ```
# Alternatively, you could use a `for` loop. As it is going to be an infinite generator, it will continue to execute until you stop it manually so DO NOT RUN IT.
# ```python
# ## DO NOT RUN IT
# gen = fib()
# for i in gen:
#     print(i)
# # 1
# # 1
# # 2
# # 3
# # 5
# # ...
# ```

# %% [markdown]
# ### Exercice 1.1: Processing long sequences

# %% [markdown]
# When working on natural languages, sentences are usually decomposed into words or "tokens". Although DNA sequences do not contain words, it is common to decompose them into k-mers (akin to n-grams in natural languages). A k-mer is a continuous sequence of k nucleotides.
#
# For example, decomposing `AACAT` into k-mers with k=3 (tri-mers) yields `[AAC, ACA, CAT]`.
#
# Create a function to decompose an input DNA sequence into k-mers of length k. The function should `yield` k-mers rather than retaining the whole (potentially very long) list in memory.

# %%
import random

random.seed(1)
N = 10
nucs = "actg"
seq = "".join(random.choices(nucs, k=10))
seq


# %%
from typing import Iterator

# Solution

def yield_kmers(seq: str, k: int = 2) -> Iterator[str]:
    """Extract k-mers of length k from input sequence
    
    >>> [k for k in get_kmers('aactc', 3)]
    ['aac', 'act', 'ctc']
    >>> type(yield_kmers('ac', 1))
    generator
    """
    for s in range(len(seq) - k + 1):
        yield seq[s : s + k].lower()


# %%
# Testing your function

kmers = []
for kmer in yield_kmers(seq):
    kmers.append(kmer)
kmers

# %% [markdown]
# Python also offers several utility functions producing iterators. For example,`zip` and `enumerate` are very useful:

# %%
# Zip combines elements from two iterables
list(zip(['a', 'b', 'c'], [1, 2, 3]))

# %%
# Enumerate yields the values and corresponding index
list(enumerate(['a', 'b', 'c']))

# %% [markdown]
# -----
# ## List comprehensions
#
# > List comprehensions provide a concise way to create lists. 
#
# Detailed instructions can be found at [Python documentation](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions).

# %%
# This example generates a list of 3 sequencess using a list comprehension
seq_list = ["".join(random.choices(nucs, k=10)) for _ in range(3)]

# %% [markdown]
# ### Exercice 2.1: Unconditional comprehensions

# %% [markdown]
# Write the following code as a list comprehension.
# ```python
# nuc_list = []
#
# for seq in seq_list:
#     nucs = []
#     for nuc in seq:
#         nucs.append(nuc)
#     nuc_list.append(nucs)
# nuc_list
#
# # [['g', 'g', 'a', 'c', 't', 't', 'g', 'c', 'g', 't'],
# #  ['c', 't', 'g', 'g', 't', 't', 'a', 'a', 'g', 'c'],
# #  ['a', 't', 't', 't', 'c', 'c', 't', 'g', 't', 'c']]
#  ```

# %%
# Solution

nuc_list = [[nuc for nuc in seq] for seq in seq_list]
nuc_list

# %% [markdown]
# ### Exercice 2.2: Conditional comprehensions

# %% [markdown]
# The DNA double helix is made up of 2 anti-parallel polymer strands. These strands are complementary to each other: an `A` can only be paired with a `T` on the opposite strand, and a `C` with a `G`. For example, a segment of DNA could look like this:
#
# -- `AAGCT`-->
#
# <-`TTCGA`--
#
# The strands are antiparallel, meaning that any region will be read in the opposite direction on the other strand. Therefore, a biological sequence on one strand can be reverse-complemented to obtain the sequence read on the other strand.
#
# Below is a helper function to return the reverse-complement of an input DNA sequence.
#

# %%
def rev_comp(seq: str) -> str:
    """Returns the reverse-complement of a DNA sequence

   >>>rev_comp('aaagc')
   'gcttt'
   """
    comp_map = seq.maketrans("acgt", "tgca")
    comp_seq = seq.translate(comp_map)
    rev_comp = comp_seq[::-1]
    return rev_comp


# %% [markdown]
# Each k-mer $s$ possesses an equivalent reverse-complement $\bar s$, which is redundant in some applications. It is frequent to use only the _canonical_ sequence of a k-mer pair, this is the smaller of the two reverse complementary sequences according to a function $h$.
#
# ${\rm canonical}(s|h)=\left\{\begin{array}{ll}
# s & \mbox{if $h(s)<h(\overline{s})$}\\
# \overline{s} & \mbox{otherwise}\\
# \end{array}\right.$
#
# Use a list comprehension to generate the list of all possible non-reverse complementary 4-mers, selecting the first in lexicographic order for each pair (e.g. `AAC` and not `GTT`). In other words, the canonical k-mer should be the one that comes earliest in alphabetical order.
#
# > Hint: The `itertools` package, shipped with the python standard library, contains useful functions to generate all combinations of elements.

# %%
import itertools

# Solution

[k for k in map("".join, itertools.product("acgt", repeat=4)) if k < rev_comp(k)]

# %% [markdown]
# ### Exercice 2.3: Compute k-mer distance between 2 sequences

# %% [markdown]
# K-mer frequencies can reflect evolutionary distances between species. A simple way to compare similarity between 2 DNA sequences is to measure a total distance between their k-mer frequencies.
#
# Write a function to compute the euclidean distance between the canonical k-mer frequencies of a pair sequences. You can re-use the `yield_kmers` and `rev_comp` functions to extract canonical k-mers from a sequence.
#
# $D_{i,j}=\sqrt{\sum_{k \in V}{(P_i(k) - P_j(k))^2}}$
#
# Where $i$ and $j$ are the input sequences, $V$ is the set k-mers and $P_i(k)$ is the probability of occurence of k-mer $k$ in sequence $i$ (i.e. The proportion of counts for k-mer $k$).
#
# > Tip: The `defaultdict` structure, from the `collections` module in the standard library, can be convenient here.
#

# %%
seq1 = "aact"
seq2 = "aatc"

# %%
from typing import Dict
from collections import defaultdict
import numpy as np

# Solution

def count_kmer_freqs(seq: str, k: int = 2) -> Dict[str, float]:
    """Compute the occurrences of each canonical k-mer of
    length k in seq.
    
    >>>count_kmer_freqs('tcagg', 2)
    defaultdict(float, {'ga': 0.25, 'ca': 0.25, 'ag': 0.25, 'cc': 0.25})
    """
    freqs = defaultdict(float)
    n_kmers = len(seq) - k + 1
    # Extract k-mers from input sequence
    for kmer in yield_kmers(seq, k=k):
        cano = min(kmer, rev_comp(kmer))
        freqs[cano] += 1 / n_kmers
    return freqs


def get_kmer_dist(seqi: str, seqj: str, k: int = 4) -> float:
    """Compute the euclidean distance between the
    sum of canonical k-mer frequencies of 2 sequences.
    
    >>> get_kmer_dist('aact', 'aatc', k=3)
    1.0
    """
    dist = 0
    freqi = count_kmer_freqs(seqi, k=k)
    freqj = count_kmer_freqs(seqj, k=k)
    kmers_union = freqi.keys() | freqj.keys()
    for kmer in kmers_union:
        dist += (freqi[kmer] - freqj[kmer]) ** 2
    dist = np.sqrt(dist)
    return dist


# %% [markdown]
# The cell below will test your function on gene sequences from 3 different species: Human, cat and the bacterium _E. coli_.
#
# K-mer frequencies from humans and cats should be more similar to each other, compared to _E. coli_.

# %%
# Test your function on (Real) DNA sequences
seqs = []
with open("../data/mystery_genes.csv") as file_handle:
    for line in file_handle:
        seqs.append((line.strip().split(",")))

# Distance computed between each pair of sequences
for sa, sb in list(itertools.combinations(seqs, 2)):
    dist = get_kmer_dist(sa[1], sb[1])
    print(f"{sa[0]} vs {sb[0]}: {dist}")

# %% [markdown]
# ## Lambda Operator and the functions map() and reduce()

# %% [markdown]
# ### Exercise 3.1: Transform and concatenate sequences
#
# Given a list of mixed-case characters, use map and reduce to transform them to lowercase and concatenate them in one line of code.

# %%
from functools import reduce

mixed_case = ["".join(random.choices("actgACTG", k=10)) for _ in range(5)]
mixed_case

# %%
# Solution
reduce(lambda a, b: a + b, map(str.lower, mixed_case))

# %% [markdown]
# ### Exercise 3.2: Approximate the exponential function

# %% [markdown]
# Approximate the Maclaurin series for the exponential function at X with a polynomial of degree K
#
# Please use the [map()](https://docs.python.org/3/library/functions.html#map) & [reduce()](https://docs.python.org/3/library/functools.html#functools.reduce) functions. For example:
# ```python
# l1 = [1, 2, 3, 4, 5]
# l2 = map(lambda x: x + 1, l1) # l2 is a map object which is an iterator
# list(l2)
# # [2, 3, 4, 5, 6]
#
# from functools import reduce
# reduce(operator.add, l1) # which calculates ((((1+2)+3)+4)+5)
# # 15
# ```

# %%
from functools import reduce
from math import factorial
import operator

K = 10
X = 4


def evaluate_polynomial(a, x):
    # Write your code here
    # xi: [x^0, x^1, x^2, ..., x^(K-1)]
    # axi: [a[0]*x^0, a[1]*x^1, ..., a[K-1]*x^(K-1)]
    # return sum of axi
    return 0


# %%
# Solution
from functools import reduce
from math import factorial
import operator

K = 10
X = 4


def evaluate_polynomial(a, x):

    xi = map(lambda i: x ** i, range(0, len(a)))  # [x^0, x^1, x^2, ..., x^(K-1)]
    axi = map(operator.mul, a, xi)  # [a[0]*x^0, a[1]*x^1, ..., a[K-1]*x^(K-1)]
    return reduce(operator.add, axi, 0)  # sum of axi


# %%
evaluate_polynomial([1 / factorial(x) for x in range(0, K)], X)

# %% [markdown]
# ## Lists of characters and the functions join() and append()

# %% [markdown]
# ### Exercice 4.1: Smart password generator

# %% [markdown]
# Write a function for generating a password of $N$ characters. Your password must have a **fair** number of lowercase letters, uppercase letters, and numbers. For example, if $N = 8$, you start filling out the list with 2 lowercase letters, 2 uppercase letters and 2 numbers. Then you complete the list with $N - 3*2$ lowercase letters.

# %% [markdown]
# Please use of the join(), append() and random.shuffle() functions.

# %%
import random

lowercase_letters = "abcdefghijklmnopqrstuvwxyz"
uppercase_letters = lowercase_letters.upper()

N = 8
password = []

# Add your code here

print(password)

# %%
# Solution
import random

lowercase_letters = "abcdefghijklmnopqrstuvwxyz"
uppercase_letters = lowercase_letters.upper()

N = 8
password = []

for i in range(N // 3):
    password.append(lowercase_letters[random.randrange(len(lowercase_letters))])
    password.append(uppercase_letters[random.randrange(len(uppercase_letters))])
    password.append(str(random.randrange(10)))

for i in range(N - len(password)):
    password.append(lowercase_letters[random.randrange(len(lowercase_letters))])

random.shuffle(password)
password = "".join(password)

print(password)

