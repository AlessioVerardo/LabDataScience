# -*- coding: utf-8 -*-
# %% [markdown]
# # Pandas and scikit-learn

# %% [markdown]
# ## Exercise 1: Getting familiar with pandas
#
# Pandas is another package providing vectorized operations. It makes data wrangling easier by providing a dataframe structure with column names and indices.
#

# %%
import pandas as pd
import numpy as np
import random

# First step: make the data frame
dates = pd.date_range('20200101', '20201231') #366
data = pd.DataFrame(np.random.randn(366,4), index=dates, columns=list('ABCD'))

# %% [markdown]
# ### 1.1 Inspect the Pandas DataFrame
#
# Use the following commands: head(), tail(), describe(), info()

# %%
# Solution
data.head()

# %%
data.tail()

# %%
data.describe()

# %%
data.info()

# %% [markdown]
# ### 1.2 Resampling Pandas DataFrames
#
# The index is a time series, and pandas has a build-in command for re-sampling dataframes ([documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html)) 
#
# Use resample to get the median every 2 days and save this as a `new_data` dataframe

# %%
# Solution
new_data = data.resample('2D').median()
new_data.head()

# %% [markdown]
# ### 1.3 Inspect the new (resampled) DataFrame
#
# See the difference in shape compared to the inital dataframe

# %%
# Solution
data.shape, new_data.shape

# %% [markdown]
# ### 1.4  Writing Pandas DataFrames to files
#
# Write your new dataframe to a tsv (tab-separated values) file

# %%
# Solution
new_data.to_csv('test_pythoncourse.tsv', sep='\t')

# %% [markdown]
# ### 1.5 Merge the two DataFrames.
#
# There are several ways you can do this, see also [here](https://pandas.pydata.org/pandas-docs/stable/merging.html).

# %%
# Solution
# generate another dataframe and merge them together
new_data.merge(data, how='left').head()

# %% [markdown]
# Note: the purpose of exercises 1.5 is to familiarise yourselves with the DataFrames API. But now sit back and ponder: would it make sense to do what we just did in practice 😜 ?

# %% [markdown]
# ### 1.6 DataFrames Column operations
#
# There are several ways to perform actions on the dataframe columns. The dataframe has several columns containing negative values. For this exercise, create a new column with the absolute value of one existing column. First using a list comprehension, then using a lambda function. You can use the magic `%timeit` (check [here](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit) for usage) to see if there is a difference between these operations.

# %%
# Solution
# method 1: list comprehension
# %timeit data['E'] = [ abs(x) for x in data['B'] ]

# %%
# method 2: lambda function
# %timeit data['F'] = data['B'].apply(lambda x: abs(x) ) 

# %% [markdown]
# ## Exercise 2: Supervised learning: DNA sequence classification

# %% [markdown]
# ### 2.1: Read the data
#
# We will use a dataset consisting of 106 DNA sequences. All sequences are the same length (57 nucleotides), but half of them are bacterial promoters.
#
# The input table contains 3 columns:
#
# * Whether each sequence is a promoter (+) or not (-).
# * A unique identifier
# * A DNA sequence of 57 nucleotides.
#
# The dataset is stored as a CSV file.
#
# > Note: A promoter is a regulatory DNA sequence located just before the start of a gene and modulating its activity.

# %%
data_path = '../data/promoter-sequences-classification/promoter_sequences.csv'

# %%
# Solution
seq_df = pd.read_csv(data_path, names=['promoter', 'name', 'sequence'])
seq_df

# %% [markdown]
# ### 2.2: Clean the data
#
# The DNA sequences contain unwanted characters, use [pandas string methods](https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling) on the column to address the problem.

# %%
# Solution
seq_df['sequence'] = seq_df['sequence'].str.strip()

# %% [markdown]
# ### 2.3: Extract features
#
# Use the `get_all_kmer_freqs` function provided below to extract k-mer frequencies from each sequence, and add them as columns to the dataframe.

# %%
from typing import Iterator, List
import itertools
from collections import defaultdict

def yield_kmers(seq: str, k: int = 4) -> Iterator[str]:
    for s in range(len(seq) - k + 1):
        yield seq[s : s + k].lower()


def rev_comp(seq: str) -> str:
    comp_map = seq.maketrans("acgt", "tgca")
    comp_seq = seq.translate(comp_map)
    rev_comp = comp_seq[::-1]
    return rev_comp


def get_kmers_vocab(k: int):
    prods = itertools.product("acgt", repeat=k)
    return [k for k in map("".join, prods) if k < rev_comp(k)]


# Adapted count_kmer_freqs function from the previous exercises
def get_all_kmer_freqs(seq: str, k: int = 4) -> List[float]:
    vocab = get_kmers_vocab(k)
    freqs = defaultdict(float)
    n_kmers = len(seq) - k + 1
    # Extract k-mers from input sequence
    for kmer in yield_kmers(seq, k=k):
        cano = min(kmer, rev_comp(kmer))
        freqs[cano] += 1 / n_kmers
    return [freqs[k] for k in vocab]


# %%
# Solution
k = 4
freqs = np.vstack(seq_df.sequence.apply(lambda x: get_all_kmer_freqs(x, k=k)))
features = pd.concat(
    [
        seq_df[['name', 'sequence']],
        pd.DataFrame(freqs, columns=get_kmers_vocab(k))
    ],
    axis=1,
)
features

# %% [markdown]
# Combine all resulting features into a single feature table.
#
# > Hint: You can use `numpy.reshape` to get reshape the one-hot encoded array into 2 dimensions.

# %% [markdown]
# ### 2.4: Sequence classification with scikit-learn
#
# The scikit-learn (`sklearn`) library, provides many utilities and standard models for machine learning. We will attempt to classify sequences into promoter or non-promoter using a random-forest model, as it does not suffer from the high dimensionality of our dataset and is easily interpretable.

# %%
# Train a regression model to predict promoter / non-promoter status
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# %%
X = np.array(features.drop(columns=['name', 'sequence']))
y = seq_df.promoter
clf = RandomForestClassifier()

# %% [markdown]
# Split the data into 2 sets (e.g. with `train_test_split` to train the model on the first set and check predictions on the second.

# %%
# Solution
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y)
clf.fit(X_train, y_train)
accuracy_score(y_test, clf.predict(X_test))

# %% [markdown]
# Run the cell several times, what do you observe in terms of prediction accuracy ?

# %% [markdown]
# To properly assess the model's performance, we should use a cross-validation strategy. Here, the dataset is relatively small, so we can use leave-one-out crossvalidation.
# This method involves repeatedly excluding one sample from the training set and attempting to predict it, until all samples have been tested. It is slow; if there are N samples, it involves training the model N times on N-1 samples.
#
# Implement Leave-one-out cross validation using the `sklearn` API.

# %%
# Solution
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
cv = LeaveOneOut().split(X)
preds = np.zeros(y.shape, dtype=str)
for train_idx, test_idx in cv:
    clf.fit(X[train_idx, :], y[train_idx])
    pred = clf.predict(X[test_idx, :])
    preds[test_idx] = pred
    
ConfusionMatrixDisplay.from_predictions(y, preds)
plt.title(f"LOO cross validation accuracy: {accuracy_score(y, preds):.2f}")


# %% [markdown]
# ### 2.4: Interpreting the model
#
# As mentioned previously, random forests (and tree-based models in general) are readily interpretable by measuring the importance of each feature in the decision trees.
#
# Extract feature importances from the model and visualize them. What are sequence motifs are important for prediction ? What do you think of it ? (Hint: [useful info here](https://en.wikipedia.org/wiki/Promoter_(genetics)#Elements))

# %%
# Solution
plt.figure(figsize=(10, 10))
# Refit on whole dataset
clf.fit(X, y)
imp_idx = np.argsort(clf.feature_importances_)
plt.barh(features.columns[2:][imp_idx][-10:], clf.feature_importances_[imp_idx][-10:])

# %% [markdown]
# ### 2.5: Bonus: Hyperparameter tuning with gridsearch
#
# The default model hyperparameters may be suboptimal for our dataset. In order to find the optimal combination of parameters, we can probe the model performance across parameter space. This operation is called grisearch and `sklearn` also implements a set of utility functions to facilitate the process.

# %%
import yaml
from sklearn.model_selection import GridSearchCV
# Search for optimal parameters (uses 5-fold cv for validation by default)
params = {
    "max_features": [3, 5, 10, 20, 50],
    "n_estimators": [2 ** p for p in range(6, 8)],
}
# Note n_jobs is the number of parallel jobs, this should depend on your machine's CPUs
gs = GridSearchCV(clf, param_grid=params, verbose=True, n_jobs=1)
gs.fit(X, y)
print(
    f"Random forest hyper-parameter search:\n"
    f"Obtained a best accuracy of {gs.best_score_:.2f} "
    f"with the following parameters: \n{yaml.dump(gs.best_params_)}",
)
clf.set_params(**gs.best_params_)
clf.fit(X, y)

# %% [markdown]
# ## Exercise 3: Unsupervised learning with K-means

# %% [markdown]
# ### 3.2 Principal Component Analysis
#
# Apply PCA to the K-mer frequencies. Store the first two principal components and their true cluster index (ground truth) into a new dataframe. Visualize the sequences using these principal components (as x and y axes), colored by promoter status.

# %%
# Solution
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
pca = PCA(n_components=3)
pcs = pca.fit_transform(X)
sns.pairplot(pd.DataFrame(pcs[:, :3]).assign(prom=y), hue='prom')


# %% [markdown]
# Use K-means clustering to identify 2 clusters what proportion of points are correctly assigned ?

# %%
from sklearn.cluster import KMeans
km = KMeans(2)
km.fit(X)
pred = km.predict(X)
sns.pairplot(pd.DataFrame(pcs[:, :3]).assign(prom=pred), hue='prom')

# %%
str_pred = [['+', '-'][p] for p in pred]
ConfusionMatrixDisplay.from_predictions(y, str_pred)
plt.title(f"K-means assignment vs promoter status: {accuracy_score(y, str_pred):.2f}")
