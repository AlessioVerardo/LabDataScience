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
# # Analyzing the Gutenberg Books Corpus - part 2

# %% [markdown]
# In this notebook, we will use the Gutenberg Corpus in the same form as last week. 
#
# In the [first analysis notebook](https://dslab2021-renku.epfl.ch/projects/com490-pub/w5-exercises/files/blob/notebooks/gutenberg.ipynb) we explored various RDD methods and in the end built an N-gram viewer for the gutenberg books project. Now, we will use the corpus to train a simple language classification model using [Spark's machine learning library](http://spark.apache.org/docs/2.3.2/mllib-guide.html) and Spark DataFrames.
#
# <div class="alert alert-success">
# <h3>The structure of this lab is as follows:</h3>
#
# <ol>
#     <li>initializing Spark and loading data</li>
#     <li>construction of Spark DataFrames</li>
#     <li>using core DataFrame functionality and comparisons to RDD methods</li>
#     <li>using the Spark ML library for vectorization</li>
#     <li>building a classifier pipeline</li>
# </div>

# %% [markdown]
# **Before we start:** We will be using data from git LFS. If you have not done so yet, you must fetch the LFS data in order to replace the tiny LFS pointers with their data content.

# %%
# !git lfs pull
# !git lfs ls-files

# %%
# %load_ext sparkmagic.magics

# %% [markdown]
# ## Set up and launch the Spark runtime *on this notebook*

# %%
import os
username = os.environ['RENKU_USERNAME']
server = "http://iccluster029.iccluster.epfl.ch:8998"
from IPython import get_ipython
get_ipython().run_cell_magic('spark', line="config", 
                             cell="""{{ "name":"{0}-gutenberg-part2", "executorMemory":"4G", "executorCores":4, "numExecutors":10 }}""".format(username))

# %%
get_ipython().run_line_magic(
    "spark", "add -s {0}-gutenberg-part2 -l python -u {1} -k".format(username, server)
)

# %% [markdown]
# ## Load the data
#
# **TODO**: Load the gutenberg cleaned rdd from HDFS `/data/gutenberg/rdd`.
#
# Load this as `cleaned_rdd` using [**sc.sequenceFile**](https://spark.apache.org/docs/2.3.2/api/python/pyspark.html?highlight=sequencefile#pyspark.SparkContext.sequenceFile).

# %% language="spark"
# cleaned_rdd = sc.sequenceFile('/data/gutenberg/rdd').setName('gutenberg_rdd').cache()

# %% language="spark"
# cleaned_rdd.cache().count()

# %% language="spark"
# cleaned_rdd.first()[1][:200]

# %% [markdown]
# Note that there were a few further pre-processing steps: we removed all punctuation, made the text lowercase, and replaced whitespace characters with "_".

# %% [markdown]
# ### Load in the metadata dictionary and broadcast it
#
# Just as in the previous notebook, we will load our pre-generated metadata dictionary and broadcast it to all the executors. 

# %% language="spark"
# import json
# meta = sc.textFile('/data/gutenberg/gutenberg_metadata.json').collect()[0]
# meta_j = json.loads(meta)
# meta_b = sc.broadcast(meta_j)

# %% [markdown]
# ## DataFrames
#
# A [**DataFrame**](http://spark.apache.org/docs/2.3.2/sql-programming-guide.html#creating-dataframes) is analogous to Pandas or R dataframes. They are since v2.0 the "official" API for Spark and importantly, the development of the [machine learning library](https://spark.apache.org/docs/2.3.2/ml-guide.html) is focused exclusively on the DataFrame API. Many low-level optimizations have been developed for DataFrames in recent versions of Spark, so that the overheads of using Python with Spark have also been minimized somewhat. Using DataFrames allows you to specify types for your operations which means that they can be offloaded to the Scala backend and optimized by the runtime. 
#
# However, you frequently will find that there simply is no easy way of doing a particular operation with the DataFrame methods and will need to resort to the lower-level [**RDD**](https://spark.apache.org/docs/2.3.2/rdd-programming-guide.html) API. 
#
# ## Creating a DataFrame
#
# Here we will create a DataFrame out of the RDD that we were using in the previous excercies. The DataFrame is a much more natural fit for this dataset. The inclusion of the book metadata is much more natural here, simply as columns which can then be used in queries. 
#
# To begin, we will map the RDD elements to type [**Row**](http://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html#pyspark.sql.Row) and recast the data as a DataFrame. Note that we are lazy here and are just using the default `StringType` for all columns, but we could be more specific and use e.g. `IntegerType` for the `gid` field. 

# %% [markdown]
# We define a method to create Row objects, this is because the cluster is using Python 2. With Python 3 this can be done inline:
# `cleaned_rdd.map(lambda x: Row(**meta_b.value[x[0]], text=x[1]))`

# %% language="spark"
# def createRow(book, meta):
#     dict = meta.value[book[0]]
#     dict['text'] = book[1]
#     return Row(**dict)

# %% language="spark"
# from pyspark.sql import Row
# from pyspark.sql.types import IntegerType, StringType, ArrayType, StructField, StructType
#
# # set up the Row 
# df = spark.createDataFrame(
#     cleaned_rdd.map(lambda x: createRow(x, meta_b))
# ).cache()

# %% [markdown]
# For inspection, the `Row` class can be conveniently cast into a `dict`:

# %% language="spark"
# # first row
# df.first().asDict()

# %% language="spark"
# df.columns

# %% [markdown]
# The DataFrame includes convenience methods for quickly inspecting the data. For example:

# %% language="spark"
# df.describe('birth_year').show()

# %% [markdown]
# Certain operations are much more convenient with the DataFrame API, such as [**groupBy**](https://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html#pyspark.sql.DataFrame.groupBy), which yields a special [**GroupedData**](http://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html#pyspark.sql.GroupedData) object. Check out the API for the different operations you can perform on grouped data -- here we use `count` to get the equivalent of our author-count from the previous exercise:

# %% language="spark"
# (df.groupBy('author_name')
#    .count()
#    .sort('count', ascending=False)
#    .show()
# )

# %% [markdown]
# ### Accessing columns
#
# Columns can be accessed in a variety of ways, and usually you can just pass a column name to DataFrame methods: 

# %% language="spark"
# df.select('birth_year').show()

# %% [markdown]
# However, columns are also objects, so they have methods of their own that can be useful. See [here](http://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html?highlight=isin#pyspark.sql.Column). You can access them very simply like this:

# %% language="spark"
# df.birth_year

# %% [markdown]
# ### Creating new columns
#
# Lets make a new column with a publication date similar to the previous notebook:

# %% language="spark"
# df = df.withColumn('publication_year', (df.birth_year + 40))

# %% [markdown]
# **TODO**: Show author name, title and publication year; sort by publication_year in descending order

# %% language="spark"
# df.select('author_name', 'title', 'publication_year').sort(df.publication_year.desc()).show()

# %% [markdown]
# # Language classification with Spark ML
#
# Here we will use some of the same techniques we developed in the last excercise, but this time we will use the built-in methods of the [Spark ML library](http://spark.apache.org/docs/2.3.2/api/python/pyspark.ml#) instead of coding up our own transformation functions. We will apply the N-Gram technique to build a simple language classification model. 
#
# The method is rather straightforward and outlined in [Cavnar & Trenkle 1994](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.3248&rep=rep1&type=pdf):
#
# For each of the English/German training sets:
#
# 1. tokenize the text (spaces are also tokens, so we replace them with "_")
# 2. extract N-grams where 1 < N < 5
# 3. determine the most common N-grams for each corpus
# 4. encode both sets of documents using the combined top ngrams
#
#
# ## Character tokens vs. Word tokens
# In the last notebook, we used words as "tokens" -- now we will use characters, even accounting for white space (which we have replaced with "_" above). We will use the two example sentences again:
#
#     document 1: "John likes to watch movies. Mary likes movies too."
#     document 2: "John also likes to watch football games"

# %% [markdown]
# ## SparkML feature transformers
#
# The SparkML library includes many data transformers that all support the same API (much in the same vein as Scikit-Learn). Here we are using the [**CountVectorizer**](http://spark.apache.org/docs/2.3.2/api/python/pyspark.ml.html#pyspark.ml.feature.CountVectorizer), [**NGram**](http://spark.apache.org/docs/2.3.2/api/python/pyspark.ml.html#pyspark.ml.feature.NGram) and [**RegexTokenizer**](http://spark.apache.org/docs/2.3.2/api/python/pyspark.ml.html#pyspark.ml.feature.RegexTokenizer). 

# %% language="spark"
# from pyspark.ml.feature import CountVectorizer, NGram, RegexTokenizer

# %% [markdown]
# ### Define the transformations
#
# We instantiate the three transformers that will be applied in turn. We will pass the output of one as the input of the next -- in the end our DataFrame will contain a column `vectors` that will be the vectorized version of the documents. 

# %% language="spark"
# regex_tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", gaps=False, pattern='\S')
# ngram = NGram(n=2, inputCol='tokens', outputCol='ngrams')
# count_vectorizer = CountVectorizer(inputCol="ngrams", outputCol="vectors", vocabSize=1000)

# %% [markdown]
# So lets see what this does to our test sentences:

# %% language="spark"
# test_df = spark.createDataFrame(
#     [
#         ('John likes to watch movies. Mary likes movies too.',),
#         ('John also likes to watch football games',)
#     ],
#     ['text']
# )
#
# test_df.collect()

# %% [markdown]
# **TODO** Figure out how to run the `test_df` through the two transformers and generate an `test_ngram_df`. `show()` the `text`, `tokens`, and `ngrams` columns.

# %% language="spark"
# test_ngram_df = ngram.transform(
#     regex_tokenizer.transform(test_df)
# )
# test_ngram_df.show()

# %% [markdown]
# **TODO**: Fit the `CountVectorizer` with `n=2` ngrams and store in `test_cv_model`:

# %% language="spark"
# test_cv_model = count_vectorizer.fit(test_ngram_df)

# %% language="spark"
# # inspect the constructed vocabulary
# test_cv_model.vocabulary

# %% [markdown]
# **TODO**: transform `test_ngram_df` into vectors:

# %% language="spark"
# test_cv_model.transform(test_ngram_df).select('vectors').show(truncate=False)

# %% [markdown]
# ## ML Pipelines
#
# Keeping track of these steps is a bit tedious -- if we wanted to repeat the above steps on different data, we would either have to write a wrapper function or re-execute all the cells again. It would be great if we could create a *pipeline* that encapsulated these steps and all we had to do was provide the inputs and parameters. 
#
# The Spark ML library includes this concept of [Pipelines](https://spark.apache.org/docs/2.3.2/ml-pipeline.html) and we can use it to simplify complex ML workflows.

# %% language="spark"
# from pyspark.ml import Pipeline
# cv_pipeline = Pipeline(
#     stages=[
#         regex_tokenizer,
#         ngram,
#         count_vectorizer,
#     ]
# )

# %% language="spark"
# (
#     cv_pipeline.fit(test_df)
#                .transform(test_df)
#                .show()
# )

# %% [markdown]
# This is much more concise and much less error prone! The really cool thing about pipelines is that I can now very easily change the parameters of the different components. Imagine we wanted to fit trigrams (`n=3`) instead of bigrams (`n=2`), and we wanted to change the name of the final column. We can reuse the same pipeline but feed it a *parameter map* specifying the changed parameter value:

# %% language="spark"
# # note the dictionaries added to fit() and transform() arguments
# (
#     cv_pipeline.fit(test_df, {ngram.n:3})
#                .transform(test_df, {count_vectorizer.outputCol: 'new_vectors'})
#                .show()
# )

# %% [markdown]
# ### Building a more complex pipeline
#
# For our language classification we want to use ngrams 1-3. We can build a function that will yield a pipeline with this more complex setup. Our procedure here is like this:
#
# 1. tokenize as before
# 2. assemble the ngram transformers to yield n=1, n=2, etc columns
# 3. vectorize using each set of ngrams giving partial vectors
# 4. assemble the vectors into one complete feature vector using [**VectorAssembler**](http://spark.apache.org/docs/2.3.2/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)

# %% language="spark"
# from pyspark.ml.feature import VectorAssembler
#
# def ngram_vectorize(min_n=1, max_n=1, min_df=1):
#     """
#     Use a range of ngrams to vectorize a corpus.
#     
#     Arguments:
#     
#     min_n: minimum ngram 
#     
#     max_n: maximum ngram
#     
#     min_df: minimum dataset frequency
#     
#     Returns:
#     
#     A Pipeline to yield vectorized documents using the specified ngram parameters.
#     """
#     tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", gaps=False, pattern='\S')
#     
#     ngrams = []
#     count_vectorizers = []
#     
#     for i in range(min_n, max_n+1):
#         ngrams.append(
#             NGram(n=i, inputCol='tokens', outputCol='ngrams_'+str(i))
#         )
#         count_vectorizers.append(
#             CountVectorizer(inputCol='ngrams_'+str(i), outputCol='vectors_'+str(i), vocabSize=1000, minDF=min_df)
#         )
#     
#     assembler = VectorAssembler(
#         inputCols=['vectors_'+str(i) for i in range(min_n, max_n+1)], outputCol='features')
#     
#     return Pipeline(stages=[tokenizer] + ngrams + count_vectorizers + [assembler])

# %% language="spark"
# ngram_vectorize(1,3).fit(test_df).transform(test_df).select('features').show(truncate=False)

# %% [markdown]
# ### Preparing the DataFrames and models
#
# For our language classifier we will use just two languages (English and either German or French). We need to create a DataFrame that is filtered to just include those languages. 
#
# In addition, we will need this step of transforming raw string documents into vectors when we try the classifier on new data. We should therefore save the fitted NGram model for later. 

# %% [markdown]
# **TODO:** filter the DF down to "en", "de" and "fr" languages.

# %% language="spark"
# lang_df = df.filter(df.language.isin('en', 'de', 'fr')).cache()

# %% [markdown]
# **TODO:** Construct the `ngram_model` by using `ngram_vectorize`. Once we have this model, we can simply call its `transform` method to turn any corpus of documents into feature vectors. 

# %% language="spark"
# ngram_model = ngram_vectorize(1,3, min_df=10).fit(lang_df)

# %% language="spark"
# ngram_model.transform(lang_df).select('features').first()

# %% [markdown]
# ## Building the classifier
#
# We have successfully transformed the dataset into a representation that we can (almost) feed into a classifier. What we need still is a label column as well the final stage of the pipeline that will fit the actual model. 
#
# To generate labels from the language column, we will use the [**StringIndexer**](https://spark.apache.org/docs/2.3.2/api/python/pyspark.ml#pyspark.ml.feature.StringIndexer) as a part of our pipeline. For the classification we will use the simplest possible [**LogisticRegression**](https://spark.apache.org/docs/2.3.2/api/python/pyspark.ml#pyspark.ml.classification.LogisticRegression) -- once you've convinced yourself that you know how it works, go ahead and experiment with other [classifiers](http://spark.apache.org/docs/latest/api/python/pyspark.ml#module-pyspark.ml.classification).

# %% language="spark"
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.feature import StringIndexer

# %% [markdown]
# **TODO:** Set up a `classification_pipeline`. Use the N-gram model we defined above as a starting stage, followed by a `StringIndexer` and a `LogisticRegression` classifier. Make sure you read the documentation on these!
#
# Note that we can use the pre-trained N-gram model -- the `Pipeline` will automatically infer that the stage is already complete and will only use it in the transformation step. 

# %% language="spark"
# classification_pipeline = Pipeline(
#     stages=[ngram_model, 
#             StringIndexer(inputCol='language', outputCol='label'),
#             LogisticRegression(regParam=0.002, elasticNetParam=1, maxIter=10)
#            ]
# )

# %% [markdown]
# Run the classifier! The fitting will take a while -- you may want to run this first on a subset of the data

# %% [markdown]
# **TODO:**  Take a 20% sample of lang_df (with replacement). Randomly split this sample into 80% _training_ and 20% _test_ data. Use the [**sample**](https://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html?highlight=randomsplit#pyspark.sql.DataFrame.sample) and [**randomSplit**](https://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html?highlight=randomsplit#pyspark.sql.DataFrame.randomSplit) DataFrame methods.

# %% language="spark"
# # Split the training and test sets
# training, test = lang_df.sample(True, 0.2).randomSplit([0.8,0.2])

# %% language="spark"
# classifier = classification_pipeline.fit(training)

# %% language="spark"
# # check the predictions 
# for lang in ['en', 'fr', 'de']:
#     print('Predictions for {0}'.format(lang))
#     (classifier.transform(
#         test.filter(test.language == lang))
#             .select('label', 'probability', 'prediction')
#             .show(10, truncate=False))

# %% [markdown]
# You should be seeing mostly good agreement between `label` and `prediction`.

# %% [markdown]
# ### Improving the model and continuing the exploration of the data
#
# We have completed the basic model training, but many improvements are possible. One obvious improvement is hyperparameter tuning -- check out the [docs](http://spark.apache.org/docs/2.3.2/ml-tuning.html#ml-tuning-model-selection-and-hyperparameter-tuning) for some examples and try it out!

# %% [markdown]
# Some other ideas for things you could do with this dataset: 
#
# * try other [classifiers that are included in MLlib](http://spark.apache.org/docs/2.3.2/mllib-classification-regression.html)
# * build a regression model to predict year of publication (may be better with word ngrams)
# * do clustering on the english books and see if sub-groups of the language pop up
# * cluster by author -- do certain authors write in similar ways?

# %%
