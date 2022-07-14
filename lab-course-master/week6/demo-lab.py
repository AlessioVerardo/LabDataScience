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

# %% [markdown] slideshow={"slide_type": "slide"}
# # Five minute `DataFrame` demo

# %% [markdown]
# ### 1. Initialize the `SparkSession` which allows us to use the RDD and DataFrame APIs

# %% [markdown]
# First, we need to get a Spark application. We use a sparkmagic client to connect remotely to the Spark cluster.

# %%
# %load_ext sparkmagic.magics

# %% slideshow={"slide_type": "slide"}
import os
from IPython import get_ipython
username = os.environ['RENKU_USERNAME']
server = "http://iccluster029.iccluster.epfl.ch:8998"
get_ipython().run_cell_magic('spark', line="config", 
                             cell="""{{ "name":"{0}-demo1",
                                        "executorMemory":"4G",
                                        "executorCores":4,
                                        "numExecutors":10 }}""".format(username))

# %%
# Use 
get_ipython().run_line_magic(
    "spark", "add -s {0}-demo1 -l python -u {1} -k".format(username, server)
)

# %% [markdown]
# ----
# Now let's get some information about the current sessions and Livy endpoints.

# %%
# This command gives you details about your sparkmagic sessions
# %spark info

# %% [markdown]
# You can learn more about magic commands, checkout in particular `%spark`

# %% tags=[]
# %spark?

# %% [markdown] slideshow={"slide_type": "slide"}
# ### 2. Read in some data and turn it into an RDD of tuples

# %% [markdown]
# First of all we need to initialize an `SQLContext`.
# The `SQLContext` allows us to connect the engine with different data sources. It is used to initiate the functionalities of Spark SQL.

# %% language="spark"
# from pyspark.sql import Row
# from pyspark.sql import SQLContext
#
# sqlContext = SQLContext(sc)

# %% [markdown]
# We have a small text file for our demo, it is on hdfs under `/data/lab/people.txt`

# %% [markdown]
# Now we create an DataFrame directly from the CSV file on HDFS.

# %% language="spark"
# people_df = spark.read.load("/data/lab/people.txt",
#                              format='csv',
#                              sep=',',
#                              header='false')

# %% [markdown]
# Alternatively we could read the file as an RDD, format the `RDD` into columns and convert it to a DataFrame.

# %% language="spark"
# people_rdd = sc.textFile("/data/lab/people.txt") \
#                .map(lambda line: line.split(',')) \
#                .map(lambda x: Row(name=x[0], age=int(x[1])))
#
# people_df = sqlContext.createDataFrame(people_rdd)

# %% language="spark"
# people_df.printSchema()

# %% [markdown] slideshow={"slide_type": "slide"}
# When the `DataFrame` is constructed, the data type for each column is inferred:

# %% slideshow={"slide_type": "-"} language="spark"
# people_df.first()

# %% language="spark"
# people_df.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# There are some convenient methods for pretty-printing the columns:

# %% [markdown] slideshow={"slide_type": "slide"}
# Let's compare `RDD` methods and `DataFrame` -- we want to get all the people older than 25: 

# %% language="spark"
# # using the RDD
# people_rdd.filter(lambda x: int(x[0])>25).collect()

# %% language="spark"
# # using the DataFrame
# people_df.filter(people_df.age > 25).take(10)

# %% [markdown] slideshow={"slide_type": "slide"}
# No need to write `map`s if you can express the operation with the built-in functions. You refer to columns via the `DataFrame` object:

# %% language="spark"
# # this is a column that you can use in arithmetic expressions
# people_df.age

# %% language="spark"
# people_df.select(people_df.age, (people_df.age*2).alias('x2')).show()

# %% language="spark"
# # equivalent RDD method
# people_rdd.map(lambda x: Row(age=x[0],x2=int(x[0])*2)).collect()

# %%
