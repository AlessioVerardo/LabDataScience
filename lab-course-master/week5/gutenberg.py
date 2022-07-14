# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analyzing the Gutenberg Books Corpus
#
# The [Gutenberg Project](http://www.gutenberg.org/) is a collection of free books available online. In this lab we will use it to learn how to do analysis on text-based key-value datasets. 
#
# Our ultimate goal in this lab is to produce something like the [Google Ngram Viewer](https://books.google.com/ngrams) but for the Gutenberg book corpus. 
#
# The structure of this lab is as follows:
#
# 1. basic inspection of data using Spark
# 2. using basic Spark functions like `reduceByKey` to perform aggregations - how many authors are there? How many unique words?
# 3. transformation of the data - vectorizing text documents
# 4. creation of an N-gram viewer
#
# Since we will be using spark, it is recommended to keep the [pyspark python API documentation](https://spark.apache.org/docs/2.3.2/api/python/pyspark.html) on the side for the exercises.

# %% [markdown]
# **Before we start:** We will be using data from git LFS. If you have not done so yet, you must fetch the LFS data in order to replace the tiny LFS pointers with their data content.

# %%
# !git lfs pull
# !git lfs ls-files

# %% [markdown]
# # Part I. First steps with Spark on the Gutenberg data

# %% [markdown]
# ## Starting up the Spark runtime

# %%
# %load_ext sparkmagic.magics

# %%
# %spark add -l python -s gutenberg -u http://iccluster029.iccluster.epfl.ch:8998 -k

# %%
# %spark info

# %%
# %%spark?

# %% [markdown]
# ## The data
#
# The Gutenberg corpus has already been ingested and pre-processed for you. It is saved on HDFS as an RDD of `(book_id, text)` key-value pairs. In a separate data file, we saved a dictionary which contains the metadata for all the books in the RDD. 

# %% [markdown]
# ## Load the data from HDFS
#
# Note: the next commands will run on the remote spark driver.

# %% language="spark"
# gutenberg_rdd = sc.sequenceFile('/data/gutenberg/rdd').setName('gutenberg_rdd').cache()

# %% [markdown]
# Lets quickly have a look at what this data looks like. These inspection methods are critical for getting a handle of the data in Spark - you will use them often!

# %% language="spark"
# gutenberg_rdd.first()

# %% [markdown]
# No surprises there - the first element of the tuple is an ID string and the second element is the book text. 
#
# How many books do we have total?

# %% language="spark"
# gutenberg_rdd.count()

# %% [markdown]
# Check the line a few cells above where we loaded the data from HDFS - notice the `cache()` at the end. This means that we asked Spark to try and put as much of the RDD as possible into memory. It's not always possible to store all the data in memory, in which case Spark will periodically drop parts of RDDs and need to recompute or reload them. 
#
# * Check the Spark UI (find the link in the third `code` cell of this notebook) and find the `Storage` tab 
# * What fraction of the `gutenberg_rdd` is cached? 

# %% [markdown]
# Notice another piece of information on the storage tab -- `Cached Partitions`. The RDD is split into "partitions", each partition consisting of many pieces of data in this case the `(ID, text)` tuples. Spark distributes computation by distributing the partitions among the executors.
#
# ![spark-rdd](./figs/spark-rdd.png)

# %% [markdown]
# The "driver" in this case is our remote Spark-entry point. We connect to it from this (local) notebook via the Livy service, not shown on the picture -- it communicates with the workers via the `SparkContext` and the RDD methods. 

# %% [markdown]
# ### Load in the metadata dictionary and broadcast it
#
# The metadata is saved as a JSON file on the spark server. If we want to use the metadata along with the RDD, we need to ship the metadata to the workers as well. Spark will happily send any variable from the driver to the workers in order to satisfy the requirements of a computation, but this is wasteful if it has to be done repeatedly. The best way to achieve this is via "broadcasting" - the metadata lookup table (dictionary in this case) is sent to the workers via a torrent mechanism and stored in each worker's RAM. There, it can be quickly retrieved for computation. See [the documentation on broadcast variables](https://spark.apache.org/docs/2.3.2/rdd-programming-guide.html#broadcast-variables) and how they can be used before continuing. 
#
# One thing to remember is that each worker holds *all* of the broadcast *metadata*, but only a subset of the RDD *data*.

# %% language="spark"
# cfg = sc.getConf()
# cfg.set("spark.driver.memory", '6g')

# %% [markdown]
# -----
# **Step 1:** Loading the metadata on the driver: the file `gutenberg_metadata.json` is relatively small (about 8MB) and we already copied it to the server with `hdfs dfs -copyFromLocal...`. Since this is a regular textfile with a single line, we can use spark's textFile method to read the entire file as a string.
#
# > Note: textFile returns a sequence of lines. You have to retrieve the first line to extract its content.

# %% language="spark"
# import json
# meta = sc.textFile('/data/gutenberg/gutenberg_metadata.json').collect()[0]

# %% [markdown]
# -----
# We convert the data on the remote Spark driver, from JSON string to a dictionary object and verify its content.

# %% language="spark"
# import json
# meta_j = json.loads(meta)

# %% [markdown]
# The json object `meta_j` is just a normal python dictionary, keyed by book `gid`. For example, the metadata for the book with `gid` 101 looks like:

# %% language="spark"
# meta_j['101']

# %% [markdown]
# -----
# **Step 2:** The metadata is loaded on the Spark driver, we now ship the metadata to all of the workers.

# %% language="spark"
# meta_b = sc.broadcast(meta_j)

# %% [markdown]
# Remember that our `gutenberg_rdd` contains `gid`'s (internal book IDs) as keys and text as values and if we want some other piece of metadata, we can just access it via the dictionary. Extract the first name of the author of the first book in the corpus below. Remembering to use `meta_b.value` to access the actual metadata dictionary, i.e. to get the metadata for `gid=101` you can do `meta_b.value['101']`.

# %% language="spark"
# # TO-DO

# %% [markdown]
# ### Cleaning up the dataset with simple filtering
#
# Later on we will want to do some aggregations based on author names and years - we therefore need to exclude any kind of "bad" data, e.g. anything with a not defined value. 
#
# Below, define a filter function that filters out all books with `None` in any of the fields `title`, `first_name`, `last_name`, or `birth_year`. Also exclude a book if `birth_year` or `death_year` include the string "BC". Then construct a `filtered_rdd` that is the result of applying the filter to the `gutenberg_rdd`.

# %% language="spark"
# # TO-DO
# def filter_func(gid) : 
#     ...     
#                                                                                                        

# %% language="spark"
# filtered_rdd = gutenberg_rdd.filter(lambda x : filter_func(x[0]))

# %% [markdown]
# ### Inspecting the metadata some more
#
# Lets do a couple more checks and practice using the Spark API. 
#
# #### How many unique authors are there in the dataset? 
#
# 1. make `author_rdd` that is composed of a string `"last_name, first_name"` (use the broadcast variable `meta_b` to get the data for each `gid`)
# 2. keep only the unique author strings (*hint*: look at the [Spark RDD API](https://spark.apache.org/docs/2.3.2/api/python/pyspark.html#pyspark.RDD) to find an appropriate method)
# 3. count the number of elements remaining

# %% language="spark"
# meta_b.value['101']

# %% language="spark"
# # TODO: map filtered_rdd to contain the string "last_name, first_name" 
# # note: make sure you use strings here and not lists of strings for author names
# # note: make sure that strings are properly encoded (use u"...")

# %% language="spark"
# # TODO: use RDD methods to obtain the distinct author name strings and count them

# %% language="spark"
# assert(n_authors == 7017)

# %% [markdown]
# #### Most-represented authors in the corpus: 
#
# Here we will use a very common map/reduce pattern to count the number of occurrences of a key. The trick is to transform the RDD into key-value pairs of `(key, 1)`. For example, we might have an RDD like:
#
#     [('a', 1),
#      ('b', 1), 
#      ('c', 1), 
#      ('a', 1), 
#      ('b', 1)]
#
# If we fed this RDD to `reduceByKey` and used the simple `add` operator (see the spark intro notebook) we would get back something like
#
#     [('a', 2), ('b', 2), ('c', 1)]
#
# 1. use the `author_rdd` from above
# 2. use the pattern `(key, 1)` to set up an RDD that can be passed to [`reduceByKey`](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.reduceByKey)
# 3. run `reduceByKey`, yielding an RDD composed of `(author, count)` tuples
# 4. sort by descending order of number of books per author and print out the top 20 (try using [`takeOrdered`](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.takeOrdered))

# %% language="spark"
# # TODO: generate a list of top 20 authors, reverse-sorted by the number of books they have in the corpus

# %% [markdown]
# Finally, lets do the same thing per language, just to get an idea of how much data there is: 

# %% language="spark"
# # TODO: generate a lang_rdd that contains just the language of each book
# lang_rdd = filtered_rdd.map(...)

# %% language="spark"
# # TODO: reduce the `lang_rdd` to yield the number of books in each language

# %% [markdown]
# ## How many unique words were used in English in these 500+ years? 
#
# We could have done the above metadata gymnastics without ever invoking a distributed processing framework by simply extracting the strings from the metadata dictionary of course -- nevertheless we used the metadata to have a closer look at some of the RDD methods. However, the actual text of each data element is where the bulk of the data volume lies. We will now start to get a feeling for the text data by first constructing a corpus-wide vocabulary. 
#
# To achieve this, we have to deconstruct each document into a list of words and then extract the unique words from the entire data set. If our dataset fits into memory of a single machine, this is a simple `set` operation. But what if it doesn't? 
#
# We'll assume the latter is the case and construct one global RDD of words. Here we aren't necessarily interested in preserving the provenance of words, but just finding the unique words in the whole corpus, so we drop the metadata altogether. 
#
# ### The steps are as follows:
#
# 1. remove punctuation from the text (a `remove_punctuation` function has been created for you below)
# 2. make the text **lowercase** (use the `lower()` method of python strings)
# 3. split the text into individual words
# 4. map the entire RDD of text into an RDD of single words (you need to use [flatMap](http://spark.apache.org/docs/2.3.2/api/python/pyspark.html#pyspark.RDD.flatMap) -- this returns a different number of elements than it takes in)
# 5. use the `distinct` method of the resulting RDD to transform it into an RDD with only unique words
#
# Here's a simple illustration of how **`flatMap`** differs from **`map`**:
#
# ![flatMap](./figs/flatMap_example.svg)

# %% [markdown]
# <div class="alert alert-info">
# <p><strong>Hint</strong></p> 
# <p>In python, splitting a string into a set of words separated by spaces is easy: </p>
# </div>

# %% language="spark"
# line = 'splitting a string is super simple'
# line.split()

# %% language="spark"
# # here we define a function that strips out all punctuation
# import re
# import string
#
# def remove_punctuation(text): 
#     """Return text without punctuation"""
#     regex = re.compile('[%s ]+' % re.escape(string.punctuation))
#     return regex.sub(' ', text)

# %% language="spark"
# # for example:
# remove_punctuation('This! Is, A; _Punctuated- Sentence/')

# %% language="spark"
# # TODO: define `english_rdd` which should contain only english-language books -- we will need this again later
# english_rdd = filtered_rdd.filter(...)

# %% language="spark"
# # TODO: define `distinct_rdd` which should contain all lowercase, unique words in the english books of the corpus
# distinct_rdd = english_rdd.flatMap(...)
# nwords = distinct_rdd.count()
# print("Number of unique English words: ", nwords)

# %% language="spark"
# distinct_rdd.take(20)

# %% [markdown]
# Of course not all of these are actual words, this is just how many character sequences separated by spaces we found. The pre-processing steps are less than perfect so there is some garbage in there. We will trim this down to just the most commonly-used ones later and that will get rid of most of the nonsense. 

# %% [markdown]
# ## What are the most common words? 
#
# A "map/reduce" tutorial has to include a word counting example -- it's basically the equivalent of a "Hello World!" in programming!
#
# So, lets count the occurences of all the words across the entire corpus. This is a fairly straightforward operation, but it exposes some very common patterns that can be useful for many tasks. To simplify this a bit, we'll use only the English-language corpus for the moment.
#
# Here are the steps we need to take:
#
# 0. keep only the english language books (use a filter)
# 1. `flatMap` each document into (`word, count`) pairs, but only for words that are not in the `stop_words` set (try with a list comprehension!)
# 2. call `reduceByKey` to sum up all the `count`s for each word
# 3. finally, sort it in descending order to see the most common words first
#
# The first part here (filtering and `flatMap`) is much like what we did before, but with a twist: for each word, check that it is *not* a member of the `stop_words` set. "Stop words" include common words like "a, the, he" etc.  

# %% language="spark"
#
# import pickle
# from io import BytesIO
# stop_words = sc.binaryFiles('/data/gutenberg/stop_words.dump').values().map(lambda p: pickle.load(BytesIO(p)))
# stop_words = stop_words.first().union(['gutenbergtm', 'gutenberg', 'electronic', 'foundation', 'license', 'copyright', 'donation', 'donations'])
# stop_words

# %% [markdown]
# Broadcast from Spark driver to the worker nodes

# %% language="spark"
# stop_words_b = sc.broadcast(stop_words)

# %% [markdown]
# You can access the frozenset from stop_words_b.value

# %% language="spark"
# 'gutenberg' in stop_words_b.value

# %% language="spark"
# # TODO: use flatMap to extract the words from each document's text using the english_rdd we made above
# #       Don't forget to make everything lowercase, remove the punctuation, and filter out the stop words!
#
# words_rdd = english_rdd.flatMap(...)

# %% [markdown]
# Now that we have our "flattened" data set, do the counting by:
#
# 1. mapping each word in `words_rdd` into a `(word, 1)` tuple 
# 2. using `reduceByKey` to calculate the word frequencies 
# 3. using the `sortBy` method to sort the word counts in descending order

# %% language="spark"
# # TODO: do the word count!
# # Note that we subsample the dataset a bit in the first step here to make the calculation a little faster
# from operator import add
# word_count = (words_rdd.sample(False, 0.2)
#                        .map(...)
#                        .reduceByKey(...)
#                        .sortBy(...)
#                        .cache())

# %% [markdown]
# ### Top 100 most common words

# %% language="spark"
# word_count.take(100)

# %% [markdown]
# # Part II. Computing word frequency vs. time
#
# Now we have all the components to build a simple tool that visualizes the relative word frequency as a function of time in the Gutenberg corpus. For inspiration, see the [Google Ngram viewer](https://books.google.com/ngrams).
#
# The next few sections require some effort to implement the first time around so most of it is filled in for you - still, try to read the code and understand what is happening, especially with regards to the aggregations that are carried out. Remember that you can always use methods like `first` or `take` to inspect the intermediate results. 
#
# ### Converting documents into vectors
#
# To make quantitative analysis of the corpus possible, we will convert each document into a vector that represents the word counts for each word appearing in the document. 
#
# This will look something like this. Imagine we have have a corpus consisting of two "documents"
#
#     document 1: "John likes to watch movies. Mary likes movies too."
#     document 2: "John also likes to watch football games"
#     
# Then our corpus vocabulary (of 1-grams) is
#
#     ["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games"]
#     
# Since this is an array and each word in the array has a unique index, we can "encode" the two documents using this index mapping. Our corpus now looks like this: 
#
#     document 1: [1, 2, 1, 1, 2, 1, 1, 0, 0, 0]
#     document 2: [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
#
# This representation means that the word "John" in the first document (index `0` in the vocabulary list) appears once, but the word "likes" (index 1 in the vocabulary) appears twice etc. This is called a "bag of words representation - see [the wikipedia explanation](https://en.wikipedia.org/wiki/Bag-of-words_model), from which the example sentences above were shamelessly taken.
#
# ### Generating word counts 
#
# Once each document is converted to a vector, doing the word counts for sub-groups of documents is a simple vector addition operation. In our case, we will reduce the vectors by year, yielding an RDD that will have the total number of occurrences of each word in every year. From there, it is trivial to look up the desired word and plot the relative frequency vs. year. 
#
# ## Create the vocabulary lookup table
# Create a look-up table of words by attaching a unique index to each word. The Spark API provides a [zipWithIndex](http://spark.apache.org/docs/2.3.2/api/python/pyspark.html#pyspark.RDD.zipWithIndex) method that makes this easy. 
#
# Above, we created the `word_count` RDD that already contains the unique words and their associated counts. To reduce the size of the lookup table (and the size of the vectors), we will restrict ourselves to using only the first 100k words. 
#
# You can either create the `(word, index)` pairs in the RDD and collect the top 100,000 as a dictionary using `collectAsMap`, or you can collect the top 100,000 words from `word_count` RDD and turn them into a dictionary locally. 
#
# These are the steps we need to take to make the vocabulary lookup from an RDD:
#
# 1. use `zipWithIndex` on the keys of `word_count` to generate a unique index for each word -- we don't care about the counts anymore, so we can get rid of the values and just work with the keys using the `keys()` method
# 2. use filter to retain only the first 100000 words 
# 3. finally, use [collectAsMap](http://spark.apache.org/docs/2.3.2/api/python/pyspark.html#pyspark.RDD.collectAsMap) to return the resulting RDD to the driver as a dictionary. 

# %% language="spark"
# # TODO: create word_lookup, a dictionary that maps each of the top 100,000 words to a unique integer
# word_lookup = word_count.keys().zipWithIndex().filter(...).collectAsMap()

# %% language="spark"
# print(len(word_lookup))

# %% [markdown]
# Make a `word_lookup` into a broadcast variable so we can use it on all the workers:

# %% language="spark"
# word_lookup_b = sc.broadcast(word_lookup)

# %% [markdown]
# This dictionary is approximately 6 Mb in size - without a broadcast, it would get sent over the network to each task, resulting in a lot of network traffic! As a broadcast variable, it gets sent only *once* to each *executor*. 

# %% [markdown]
# ## Vectorize the documents
#
# Now that we have a vocabulary lookup table, we can use this to turn each document into a vector. 
#
# This is done by counting up the occurrences of all words in the document that are also in the global vocabulary. 
#
# The function `vectorize_doc` below accomplishes this by using a dictionary to keep track of the local word count. Once the counting is done we use the counts to create a sparse vector that represents the document. A sparse vector consists of two arrays, one representing the *locations* of the non-zero values, and the other the values themselves. 
#
# To return to our contrived example from above, we had 
#
#     document 1: "John likes to watch movies. Mary likes movies too."
#     document 2: "John also likes to watch football games"
#     
# which turned into 
#
#     document 1: [1, 2, 1, 1, 2, 1, 1, 0, 0, 0]
#     document 2: [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
#
# with a vocabulary of 
#
#     ["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games"]
#     
# As sparse vectors, these two documents could be represented with two arrays: 
#
#     document 1: indices = [0,1,2,3]; values = [1, 1, 1, 1]
#     document 2: indices = [1,2,4,5,6]; values = [1, 1, 1, 1, 1]
#
# We use Spark's own `SparseVector` data type, for which we must specify a size (total number of features), a (sorted) array of indices, and an array of values. This means that we start to save space if sparsity is > 50%. Note that the `SparseVector` provides some nice higher-level methods, but it does not allow simple operations like addition. If a lot of vector arithmetic is needed, you should use the scipy sparse types instead. 

# %% [markdown]
# In the next cell, we define two functions: 
#
# * **`extract_ngrams`** converts a sequence of words or characters into a sequence of n-grams (here we are just using single worde, i.e. 1-grams so we'll postpone talking about ngrams until later)
#
# * **`vectorize_doc`** converts a document into a sparse vector by using `extract_ngrams` to tokenize it and a vocabulary mapping to turn each word into a vector component

# %% language="spark"
# import numpy as np
# from pyspark.mllib.linalg import SparseVector
#
# def extract_ngrams(tokens, ngram_range=[1,1], select_ngrams = None, ngram_type='word'):
#     """
#     Turn tokens into a sequence of n-grams 
#
#     Arguments:
#
#         tokens: a list of tokens
#
#     Optional Keywords:
#
#         ngram_range: a tuple with min, max ngram ngram_range
#     
#         select_ngrams: the vocabulary to use
#     
#         ngram_type: whether to produce word or character ngrams
#
#     Output:
#
#     Generator yielding a list of ngrams in the desired range
#     generated from the input list of tokens
#
#     """
#     join_str = "" if ngram_type=='character' else " "
#     
#     # handle token n-grams
#     min_n, max_n = ngram_range
#     n_tokens = len(tokens)
#     
#     for n in range(min_n, min(max_n + 1, n_tokens + 1)):
#         for i in range(n_tokens - n + 1):
#             if n == 1: 
#                 res = tokens[i]
#             else : 
#                 res = join_str.join(tokens[i: i+n])
#            
#             # if we are using a lookup vocabulary, check for membership here
#             if select_ngrams is not None : 
#                 if res in select_ngrams: 
#                     yield res
#             else : 
#                 yield res
#             
#
# def vectorize_doc(doc, vocab, ngram_range = [1,1], ngram_type='word') : 
#     """
#     Returns a vector representation of `doc` given the reference 
#     vocabulary `vocab` after tokenizing it with `tokenizer`
#     
#     Arguments: 
#         
#         doc: a sequence of tokens (words or characters)
#         
#         vocab: the vocabulary mapping
#         
#     Keywords:
#     
#         ngram_range: the range of ngrams to process
#         
#         ngram_type: whether to produce character or word ngrams; default is 
#         
#     Returns:
#     
#         a sparse vector representation of the document given the vocabulary mapping
#     """
#     from collections import defaultdict
#     from scipy.sparse import csr_matrix 
#         
#     # count all the word occurences 
#     data = np.zeros(len(vocab))
#     
#     for ngram in extract_ngrams(doc, ngram_range, vocab, ngram_type) : 
#          data[vocab[ngram]] += 1
#             
#     # only keep the nonzero indices for the sparse representation
#     indices = data.nonzero()[0]
#     values = data[indices]
#     
#     return SparseVector(len(vocab), indices, values)

# %% [markdown]
# Using these functions to vectorize our two-sentence test corpus: 

# %% language="spark"
# s1 = "john likes to watch movies mary likes movies too"
# s2 = "john also likes to watch football games"
# vocab = ["john", "likes", "to", "watch", "movies", "mary", "too", "also", "football", "games"]
# vocab_dict = {word:ind for ind, word in enumerate(vocab)}

# %% language="spark"
# print(s1)
# print(vectorize_doc(s1.split(), vocab_dict))
# print(s2)
# print(vectorize_doc(s2.split(), vocab_dict))

# %% [markdown]
# Now we have all the components we need to create an RDD of english-language books vectorized using the most common 100k words. All we need to do is to use `mapValues` to map the text of each document in `english_rdd` into a vector using `vectorize_doc` and our broadcast vocabulary lookup table `word_lookup_b`. Recall that the broadcast variable, `word_lookup_b` is just a wrapper for the lookup table; to pass the actual lookup table to the `vectorize_doc` function, use `word_lookup_b.value`.

# %% language="spark"
# # TODO
# vector_rdd = english_rdd.mapValues(lambda text: ...).cache()

# %% language="spark"
# vector_rdd.take(10)

# %% language="spark"
# # make sure the transformation can be carried out for all elements by using count
# # this is a good way of catching anomalies in the data
# vector_rdd.count()

# %% [markdown]
# ## Perform the aggregation
#
# We now have the entire Gutenberg English book corpus in the form of sparse vectors encoding the most used 100k words. 
#
# To get the yearly sums, we will turn the metadata of each document into its publication year (i.e. the key will be the year, the value is the vector) and then do an aggregation by year. 
#
# We will use the powerful [`treeAggregate`](http://spark.apache.org/docs/latest/api/python/pyspark.html?highlight=values#pyspark.RDD.treeAggregate) method, which requires that we specify three different components:
#
# 1. the starting aggregate
# 2. a function that adds a new value to the aggregate 
# 3. a function that adds together two aggregates
#
# The way `treeAggregate` works is that it performs the reduction in a tree pattern in order to minimize the strain on the driver. In a "normal" reduction, the workers send their results to the driver, which is tasked with putting it all together -- however, if these partial results are large (as is potentially the case here) then the driver can run into memory issues. Furthermore, most of the cluster is sitting idle while the driver performs the aggregation. `treeAggregate` fixes this by performing partial aggregations on the workers and only sending the final stages to the driver. See this old [blog post](https://databricks.com/blog/2014/09/22/spark-1-1-mllib-performance-improvements.html) for a bit more description of this method. 
#
# The aggregation methods are powerful because the "aggregate" can be any object -- we can write a class that gets passed around to do the aggregation, for example. Aggregation methods are more general reduction methods because they allow you to change the type of the variables -- in our case here, we are converting the data `(year, vector)` tuples into a dictionary of arrays. 
#
# Below, we will use an instance of a dictionary as the aggregation object and define two functions that will do the actual aggregation. 

# %% language="spark"
# def add_vector(d, data_tuple) : 
#     """Add a new vector to the aggregation dictionary
#     
#     The vectors in the aggregation dictionary are dense since for most years we can expect that 
#     this will be the case anyway. Note that we use 32-bit floats to save a bit on memory. 
#     
#     Arguments: 
#         d: the aggregation dictionary
#         
#         data_tuple: the (year, vec) tuple
#         
#     Returns: 
#         the updated aggregation dictionary 
#     """
#     # expand the data tuple
#     year, vec = data_tuple
#     
#     if year in d : 
#         
#         d[year][vec.indices] += vec.values
#     else :
#         # this is the first time we've encountered this year --> make an empty vector 
#         new_vec = np.zeros(vec.size, dtype=np.float32)
#         
#         # now put in the contents of the current vector
#         new_vec[vec.indices] = vec.values
#         
#         # create the year in the dictionary
#         d[year] = new_vec
#         
#     return d
#
# def add_dicts(d1, d2) : 
#     """Add two dictionaries together
#     
#     Arguments: 
#         d1: first dictionary
#         
#         d2: second dictionary
#         
#     Returns: 
#         merged dictionaries
#     """
#     
#     # iterate through all the items in the second dictionary
#     for year, vec in d2.items() : 
#         # if this year is also in d1, add the vectors together
#         if year in d1 : 
#             d1[year] += vec
#         # if not, create a new year entry in d1
#         else : 
#             d1[year] = vec
#     return d1

# %% [markdown]
# To get the time series, we need a date of publication - we will construct a simple function below to give us a reasonable estimate and use this value as a key in an RDD of `(year, vec)` pairs:

# %% language="spark"
# from random import gauss
#
# def publication_year(gid) : 
#     """Returns the publication year for the given gutenberg id (gid)"""
#     
#     # extract the metadata dictionary for this gid
#     meta = meta_b.value[gid] 
#     
#     birth_year = int(meta['birth_year'])
#     if meta['death_year'] is None : 
#         year = birth_year + gauss(40,5)
#     else :
#         death_year = int(meta['death_year'])
#         year = max((birth_year + death_year) / 2.0, birth_year + gauss(40,5))
#
#     return min(int(year),2015)

# %% language="spark"
# year_vec = vector_rdd.map(lambda doc: (publication_year(doc[0]), doc[1]))

# %% [markdown]
# Before we perform the aggregation, we can do one final bit of optimization. Passing around dictionaries full of large arrays can get expensive very quickly. The memory footprint of our partial results will depend on how heterogeneous the years on each partition or group of partitions are: if most of the data on a partition is for the same key (year in this case) then the dictionary we create on that partition will only contain a handful of vectors.  We can control this by first partitioning the RDD in a way that groups data with the same keys onto the same partitions. 
#
# Spark provides a `partitionBy` method that does exactly this -- by default, it uses a hash function to map the keys to partitions, but you can also pass a custom partitioner if you want. If you look at the Spark UI after executing the next cell, you'll see that the partition step caused some shuffling of data, but that the aggregation itself ran very quickly and with minimal data movement. 

# %% language="spark"
# n_partitions = year_vec.getNumPartitions()

# %% language="spark"
# year_vec.first()

# %% language="spark"
# # TODO: use an empty dictionary and the two functions defined above as arguments to the treeAggregate method
# year_sums = (year_vec.partitionBy(n_partitions)
#                      .treeAggregate(...)

# %% [markdown]
# Note that `year_sums` is a single "value", in this case our aggregate dictionary containing years as keys and vectors representing cummulative word counts as values. 

# %% language="spark"
# list(year_sums.items())[0:10]

# %% [markdown]
# ## Gutenberg Project N-gram viewer
#
# Lets plot some results!
#
# Below we define a plotting function and then show how to make a plot of some interesting examples.

# %% language="spark"
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pylab as plt
# import numpy as np
#
# plt.rcParams['figure.figsize'] = (15,8)
# plt.rcParams['font.size'] = 18
# plt.style.use('fivethirtyeight')

# %% language="spark"
# def plot_usage_frequency(words, year_data, word_lookup, plot_range = [1500,2015]) : 
#     years = sorted(year_data.keys())
#     tot_count = np.array([year_data[year].sum() for year in years])
#     
#     if ',' in words:
#         words = [word.strip() for word in words.split(',')]
#     elif type(words) is not list: 
#         words = [words]
#         
#     n_words = len(words)
#     
#     plt.figure()
#     
#     for i, word in enumerate(words) : 
#         word_ind = word_lookup[word]
#         w_count = np.array([year_data[year][word_ind] for year in years])
#         
#         plt.plot(years, smooth(w_count/(tot_count-w_count)),label=word, color = plt.cm.Set1(1.*i/n_words))
#     
#     plt.xlim(plot_range)
#     plt.xlabel('year')
#     plt.ylabel('relative frequency')
#     plt.legend(loc='upper left', fontsize = 'small')
#     
#     
# def smooth(x,window_len=11,window='hanning'):
#         if x.ndim != 1:
#                 raise ValueError("smooth only accepts 1 dimension arrays.")
#         if x.size < window_len:
#                 raise ValueError("Input vector needs to be bigger than window size.")
#         if window_len<3:
#                 return x
#         if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#                 raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
#         s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
#         if window == 'flat': #moving average
#                 w=np.ones(window_len,'d')
#         else:  
#                 w=eval('np.'+window+'(window_len)')
#         y=np.convolve(w/w.sum(),s,mode='same')
#         return y[window_len:-window_len+1]

# %% language="spark"
# words = 'giveth, environment, machine'

# %% language="spark"
# plot_usage_frequency(words, year_sums, word_lookup)

# %% language="spark"
# %matplot plt

# %% [markdown]
# If you are feeling motivated, you can adapt the workflow above to work with higher-order n-grams and allow for the lookup of phrases (i.e. "world war") instead of just single words. To do this, you have to create a new `word_lookup` table and regenerate the vectors. Since single words (i.e. one-grams) will dominate, it might make sense to build separate list of top N-grams (top two-grams, top three-grams) and then merge them together into a vocabulary map. Beware that the size of the data will increase quickly for N > 1!

# %% language="spark"
# sc.stop()

# %%
