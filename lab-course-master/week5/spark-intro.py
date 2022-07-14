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
# # Basic Spark 
#
# In the lecture we discussed -- now we'll try to actually use the framework for some basic operations. 
#
# In particular, this notebook will walk you through some of the basic [Spark RDD methods](https://spark.apache.org/docs/2.3.2/api/python/pyspark.html#pyspark.RDD). As you'll see, there is a lot more to it than `map` and `reduce`.
#
# We will explore the concept of "lineage" in Spark RDDs and construct some simple key-value pair RDDs to write our first Spark applications.
#
# If you need a reminder of some of the python concepts discussed earlier, you can make use of the [python refresher notebook](python-refresher.ipynb).

# %% [markdown]
# ----
# ## Sparkmagic
#
# We will be using the Sparkmagic client in order to interact remotely with the Spark cluster.
#
# Sparkmagic works with a remote REST server for Spark, called livy, running inside the Hadoop cluster. When you run Jupyter cells using the (py)spark kernel, the kernel will automatically send commands to livy in the background for executing the commands on the cluster. Thus, the work that happens in the background when you run a Jupyter cell is as follows:
#
# * The code in the cell is run in the kernel.
# * If it is in `%%spark` mode, the kernel sends the code, serialized in an HTTP request, to livy.
# * Otherwise, the kernel executes the code locally
# * Livy executes the code on the Spark driver in the cluster.
# * If the code is regular python/scala/R code, it will run inside a python/scala or R interpreter on the Spark driver.
# * If the code includes a spark command, a spark job will be launched on the cluster to the Spark executors from the Spark driver.
# * When the python/scala/R or spark execution is finished, the results are sent back from livy to the pyspark kernel.
# * Finally, the pyspark kernel displays the result in the Jupyter notebook.
#
# ![sparkmagic](./figs/sparkmagic.png)
#
# More info about sparkmagic can be found in [jupyter-incubator/
# sparkmagic](https://github.com/jupyter-incubator/sparkmagic), see in particular the various examples.
#
# The main thing to remember, is that code running locally and code running on the Spark driver run on different
# machines, and do not share the same memory space: variables must be explicitely copied from one
# to the other by mean of the `%%spark` and `%%send_to_spark` commands.
# Also, local files from the notebook are not visibile from the spark driver - they must be copied to the cluster before the spark driver can access them. If the content of the file is reasonably small it may passed as a variable to the spark driver, using python object serialization (i.e. pickling). To complicate matter, the version of python used in the remote Spark cluster can be different than the python version of your notebook, in which case you must be careful to use data serialization methods (e.g. pickling) that both versions of python can understand.

# %% [markdown] slideshow={"slide_type": "slide"}
# ----
# ## Starting up the Spark runtime: initializing a `SparkContext` 
#
# The `SparkContext` provides you with the means of communicating with a Spark cluster. The Spark cluster in turn is controlled by a master which orchestrates pieces of work between the various executors. Every interaction with the Spark runtime happens through the `SparkContext` in one way or another. Creating a `SparkContext` is therefore the very first step that needs to happen before we do anything else.
#
# Spark veterans may be used with the following idiom:
#
# ```
# import getpass
# import pyspark
# conf = pyspark.conf.SparkConf()
# conf.setMaster('local[2]')
# conf.setAppName('spark-intro-{0}'.format(getpass.getuser()))
# sc = pyspark.SparkContext.getOrCreate(conf)
# conf = sc.getConf()
# sc
# ```
#
# However this is not necessary here, the context is automatically created under *sparkmagic*. Just remember that *sparkmagic* enables us to run a Spark application remotely, however we are adding a level of indirection between our notebook and the spark context.

# %%
# %load_ext sparkmagic.magics

# %%
# %spark add -l python -s intro -u http://iccluster029.iccluster.epfl.ch:8998 -k

# %% language="spark"
# spark

# %% language="spark"
# sc

# %% [markdown] slideshow={"slide_type": "slide"}
# Hurrah! We have a Spark Context! Now lets get some data into the Spark universe.

# %% [markdown]
# Each Spark application runs its own dedicated Web UI -- right-click (or command-click on Mac) the `Spark UI` link two cells above to get to the UI.
#
# You will find lot of nice information about the state of your Spark job, including stats on execution time of individual tasks, available memory on all of the workers, links to worker logs, etc. You will probably begin to appreciate some of this information when things start to go wrong...
#
# **Note**: this UI service is configured to support the http protocol only, however your browser will likely try to upgrade to https. If you get a connection error, try to open the URL in a private window using the http protocol.

# %% [markdown]
# ----
# ## Creating an RDD
#
# The basic object you will be working with is the Spark data abstraction called a Resilient Distributed Dataset (RDD). This class provides you with methods to execute work on your data using the Spark cluster. The simplest way of creating an RDD is by using the `parallelize` method to distribute an array of data among the executors:

# %% language="spark"
# data = range(100)
# data_rdd = sc.parallelize(data)
# print('Number of elements: ', data_rdd.count())
# print('Sum and mean: ', data_rdd.sum(), data_rdd.mean())

# %% [markdown]
# ----
# ## Map/Reduce 
#
# Lets bring some of the simple python-only examples from the [python refresher notebook](python-refresher.ipynb) into the Spark framework. The first map function we made was simply doubling the input array, so lets do this here. 
#
# Write the function `double_the_number` and then use this function with the `map` method of `data_rdd` to yield `double_rdd`:

# %% language="spark"
# def double_the_number(x) : 
#     return x*2

# %% language="spark"
# help(data_rdd.map)

# %% language="spark"
# double_rdd = data_rdd.map(double_the_number)

# %% [markdown]
# Not much happened here - or at least, no tasks were launched (you can check the console and the Web UI). Spark simply recorded that the `data_rdd` maps into `double_rdd` via the `map` method using the `double_the_number` function. You can see some of this information by inspecting the RDD debug string: 

# %% language="spark"
# print(double_rdd.toDebugString().decode())

# %% language="spark"
# # comparing the first few elements of the original and mapped RDDs using take
# print(data_rdd.take(10))
# print(double_rdd.take(10))

# %% [markdown]
# Now if you go over to check on the stages in the Spark UI you'll see that jobs were run to grab data from the RDD. In this case, a single task was run since all the numbers needed reside in one partition. Here we used `take` to extract a few RDD elements, a very very very convenient method for checking the data inside the RDD and debugging your map/reduce operations. 
#
# Often, you will want to make sure that the function you define executes properly on the whole RDD. The most common way of forcing Spark to execute the mapping on all elements of the RDD is to invoke the `count` method: 

# %% language="spark"
# double_rdd.count()

# %% [markdown]
# If you now go back to the stages page, you'll see that four tasks were run for this stage. 

# %% [markdown]
# In our initial example of using `map` in pure python code, we also used an inline lambda function. For such a simple construct like doubling the entire array, the lambda function is much neater than a separate function declaration. This works exactly the same way here.

# %% [markdown]
# Map the `data_rdd` to `double_lambda_rdd` by using a lambda function to multiply each element by 2: 

# %% language="spark"
# double_lambda_rdd = data_rdd.map(lambda n : n*2)
# print(double_lambda_rdd.take(10))

# %% [markdown]
# Finally, do a simple `reduce` step, adding up all the elements of `double_lambda_rdd`:

# %% language="spark"
# from operator import add
# double_lambda_rdd.reduce(add)

# %% [markdown]
# (Spark RDDs actually have a `sum` method which accomplishes essentially the same thing)

# %% language="spark"
# double_lambda_rdd.sum()

# %% [markdown]
# ----
# ## Filtering
#
# A critical step in many analysis tasks is to filter down the input data. In Spark, this is another *transformation*, i.e. it takes an RDD and maps it to a new RDD via a filter function. The filter function needs to evaluate each element of the RDD to either `True` or `False`. 
#
# Use `filter` with a lambda function to select all values less than 10: 

# %% language="spark"
# filtered_rdd = data_rdd.filter(lambda n : n < 10)
# filtered_rdd.count()

# %% [markdown]
# Of course we can now apply the `map` and double the `filtered_rdd` just as before: 

# %% language="spark"
# filtered_rdd.map(lambda n : n * 2).take(10)

# %% [markdown]
# Note that each RDD transformation returns a new RDD instance to the caller -- for example:

# %% language="spark"
# data_rdd.filter(lambda x: x % 2)

# %% [markdown]
# You can therefore string together many transformations without creating a separate instance variable for each step. Our `filter` + `map` step can therefore be combined into one. Note that if we surround the operations with "( )" we can make the code more readable by placing each transformation on a separate line: 

# %% language="spark"
# composite = (data_rdd.filter(lambda x: x % 2)
#                      .map(lambda x: x*2))

# %% [markdown]
# Again, if you now look at the Spark UI you'll see that nothing actually happened -- no job was trigerred. The `composite` RDD simply encodes the information needed to create it. 
#
# If an action is executed that only requires a part of the RDD, only those parts will be computed. If we cache the RDD and only calculate a few of the elements, this will be made clear:

# %% language="spark"
# composite.cache()
# composite.take(10)

# %% [markdown]
# If you look at the **Storage** tab in the Spark UI you'll see that just a quarter of the RDD is cached. Now if we trigger the full calculation, this will increase to 100%:

# %% language="spark"
# composite.count()

# %% [markdown]
# ----
# ## Key, value pair RDDs
#
# `key`,`value` pair data is the "bread and butter" of map/reduce programming. Think of the `value` part as the meat of your data and the `key` part as some crucial metadata. For example, you might have time-series data for CO$_2$ concentration by geographic location: the `key` might be the coordinates or a time window, and `value` the CO$_2$ data itself. 
#
# If your data can be expressed in this way, then the map/reduce computation model can be very convenient for pre-processing, cleaning, selecting, filtering, and finally analyzing your data. 
#
# Spark offers a `keyBy` method that you can use to produce a key from your data. In practice this might not be useful often but we'll do it here just to make an example: 

# %% language="spark"
# # key the RDD by x modulo 5
# keyed_rdd = data_rdd.keyBy(lambda x: x%5)
# keyed_rdd.take(20)

# %% [markdown]
# This created keys with values 0-4 for each element of the RDD. We can now use the multitude of `key` transformations and actions that the Spark API offers. For example, we can revisit `reduce`, but this time do it by `key`: 

# %% [markdown]
# ----
# ## `reduceByKey`

# %% language="spark"
# # use the add operator in the `reduceByKey` method
# red_by_key = keyed_rdd.reduceByKey(add)
# red_by_key.collect()

# %% [markdown]
# Unlike the global `reduce`, the `reduceByKey` is a *transformation* --> it returns another RDD. Often, when we reduce by key, the dataset size is reduced enough that it is safe to pull it completely out of Spark and into the Spark driver. A useful way of doing this is to automatically convert it to python dictionary for subsequent processing with the `collectAsMap` method:

# %% language="spark"
# red_dict = red_by_key.collectAsMap()
# red_dict

# %% language="spark"
# # access by key
# red_dict[0]

# %% [markdown]
# ----
# ## `groupByKey`
#
# If you want to collect the elements belonging to a key into a list in order to process them further, you can do this with `groupByKey`. Note that if you want to group the elements only to do a subsequent reduction, you are far better off using `reduceByKey`, because it does the reduction locally on each partition first before communicating the results to the other nodes. By contrast, `groupByKey` reshuffles the entire dataset because it has to group *all* the values for each key from all of the partitions. 

# %% language="spark"
# keyed_rdd.groupByKey().collect()

# %% [markdown]
# Note the ominous-looking `pyspark.resultiterable.Resultiterable`: this is exactly what it says, an iterable. You can turn it into a list or go through it in a loop. For example:

# %% language="spark"
# key, iterable = keyed_rdd.groupByKey().first()
# list(iterable)

# %% [markdown]
# ----
# ## `sortBy`
#
# Use the `sortBy` method of `red_by_key` to return a list sorted by the sums in descending order and print it out. 

# %% language="spark"
# sorted_red = red_by_key.sortBy(sum,ascending=False).collect()
# sorted_red

# %% language="spark"
# assert(sorted_red == [(4, 1030), (3, 1010), (2, 990), (1, 970), (0, 950)])

# %% [markdown]
# This concludes the brief tour of the Spark runtime -- we can now shut down the `SparkContex` by calling `sc.stop()`. This removes your job from the Spark cluster and cleans up the memory and temporary files on disk. 

# %% language="spark"
# sc.stop()
