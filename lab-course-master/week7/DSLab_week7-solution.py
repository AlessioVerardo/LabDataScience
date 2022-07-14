# -*- coding: utf-8 -*-
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
# # Play with PySpark

# %%
# %load_ext sparkmagic.magics

# %%
# %matplotlib inline
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = (20,15)
plt.rcParams['font.size'] = 18
plt.style.use('fivethirtyeight')

# %% [markdown]
# ## Initialize the `SparkSession`

# %%
import os
from IPython import get_ipython

# set the application name as "<your_gaspar_id>-homework3"
username = os.environ['RENKU_USERNAME']
server = "http://iccluster029.iccluster.epfl.ch:8998"

get_ipython().run_cell_magic(
    'spark',
    line='config', 
    cell="""{{ "name": "{0}-week7", "executorMemory": "4G", "executorCores": 4, "numExecutors": 10, "driverMemory": "4G"}}""".format(username)
)

# %% [markdown]
# Send `username` to Saprk kernel, which will frist start the Spark application if there is no active session.

# %%
get_ipython().run_line_magic(
    "spark", "add -s {0}-week7 -l python -u {1} -k".format(username, server)
)

# %%
# %%spark?

# %% language="spark"
# import pyspark.sql.functions as functions

# %% [markdown]
# ## PART I: `Twitter` Data

# %% [markdown]
# Load twitter data in JSON format. The twitter data is located at `hdfs:///data/twitter/json/year={year}/month={month}/day={day}/{hour}/{minute}.json.bz2`.
#
# Here, we load the data from 08:00 to 09:00 on June 1st, 2016.

# %% language="spark"
# twitter = spark.read.json('/data/twitter/json/year=2020/month=06/day=01/08')

# %% language="spark"
# twitter.show(5)

# %% [markdown]
# Check the schema of the DataFrame by
# ```python
# twitter.printSchema()
# ```
# You will get a long output as following:
# ```
# root
#  |...
#  |-- entities: struct (nullable = true)
#  |    |-- hashtags: array (nullable = true)
#  |    |    |-- element: struct (containsNull = true)
#  |    |    |    |-- indices: array (nullable = true)
#  |    |    |    |    |-- element: long (containsNull = true)
#  |    |    |    |-- text: string (nullable = true)
#  |    |...
#  |...
#  |-- id: long (nullable = true)
#  |...
#  |-- lang: string (nullable = true)
#  |...
#  |-- timestamp_ms: string (nullable = true)
#  |...
# ```

# %% [markdown]
# ### I.a: How many twitters for each language - How to plot in the context of `sparkmagic`

# %% [markdown]
# Run the cell with the magic command, which will download the dataframe in the cluster into local context:
# ```python
# %%spark -o VAR_NAME
# ```
# VAR_NAME: The Spark dataframe of name VAR_NAME will be available in the `%%local` Python context as a Pandas dataframe with the same name.

# %% magic_args="-o twitter_lang" language="spark"
# twitter_lang = twitter.filter(~functions.isnull('lang'))\
#                       .groupBy('lang').count().orderBy('count')

# %% [markdown]
# Check the dataframe object locally.

# %%
type(twitter_lang)

# %% [markdown]
# Plot the Pandas dataframe locally.

# %%
twitter_lang.plot.barh('lang', 'count', logx=True, legend='')
plt.title('Counts of Twitter for Each Language');

# %% [markdown]
# ### II.b: Build the dataframe of hashtag
#
# In homework-3, you are going to analyze the twitter hashtags and we have prepared a preprocessed dataframe for you, like:
# ```
# +-----------+------------------+----+--------------+
# |timestamp_s|                id|lang|       hashtag|
# +-----------+------------------+----+--------------+
# | 1464789600|738007191668555776|  tr|HaniMişDiploma|
# | 1464789600|738007191685373952|  en|         Wales|
# | 1464789600|738007191685373952|  en|volunteersweek|
# | 1464789600|738007191685238784|  en|         Wendi|
# | 1464789600|738007191685238784|  en|        nicole|
# |        ...|               ...| ...|           ...|
# +-----------+------------------+----+--------------+
# ```
#
# In this part, we will show you how to build such a dataframe with pyspark.

# %% [markdown]
# #### Spark SQL Data Types and Schema
# To start with, let's get ourself familiar with Spark SQL [data types](https://spark.apache.org/docs/latest/sql-reference.html#data-types) and how to specify the schema of a dataframe.
#
# While creating a Spark DataFrame we can specify the structure using [StructType](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.types.StructType) and [StructField](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.types.StructField) classes. `StructType` is a collection of `StructField`’s that defines column name, column data type, boolean to specify if the field can be nullable or not and metadata. They enable us to create complex columns like nested struct, array and map. 
#
# When we load the twitter data by `spark.read.json`, we don't need to specify the schema of the dataframe (we can also specify it). The method goes through the input once and determines the input schema for us. The schema is automatically inferred from the input structure. 
#
# ```
# root
#  |...
#  |-- entities: struct (nullable = true)
#  |    |-- hashtags: array (nullable = true)
#  |    |    |-- element: struct (containsNull = true)
#  |    |    |    |-- indices: array (nullable = true)
#  |    |    |    |    |-- element: long (containsNull = true)
#  |    |    |    |-- text: string (nullable = true)
#  |    |...
# ```
#
#
# From the schema, we learn that the entry `hashtags` is in the column `entities`, which is a nested struct column with many fields. The `hashtags` is a field of `ArrayType` which is composed of elements. It will be more clear if we show one entry of column `entities` as a RDD.

# %% language="spark"
# twitter.filter(twitter.id=="738007191685238784").select(twitter.entities).take(1)

# %% [markdown]
# If we want to extract only the `text` from the `hashtags` array, spark allows us to code as:

# %% language="spark"
# twitter.entities.hashtags.text

# %% language="spark"
# twitter.filter(functions.size(twitter.entities.hashtags.text) > 0)\
#        .select(twitter.entities.hashtags.text)\
#        .show(10)

# %% [markdown]
# #### Explode the array
#
# Once you know how to extract the hashtags of each twitter, how can you transform it into the format we want, one hashtag per row? In order to do this, please check function [explode](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.explode).
#
# #### Now, you are ready to create the dataframe:

# %% language="spark"
# twitter_hashtag = twitter.select((twitter.timestamp_ms/1000).cast('long').alias('timestamp_s'), 
#                                  twitter.id, twitter.lang, 
#                                  functions.explode(twitter.entities.hashtags.text).alias('hashtag'))\
#                          .dropna()
# twitter_hashtag.show(5)

# %% [markdown]
# ## PART II: `SBB` Data
#
# We have played with the SBB data in previous exercises and homework but mainly with `HiveQL`. Now, let's take a few examples and see how to use Spark to accomplish the same operations as in HiveQL. The full description of the data can be found in https://opentransportdata.swiss/de/cookbook/ist-daten/.
#
# We load the SBB data in __ORC__ format from `/data/sbb/orc/istdaten`. 

# %% language="spark"
# sbb = spark.read.orc('/data/sbb/orc/istdaten')

# %% language="spark"
# sbb.printSchema()

# %% language="spark"
# sbb.show(5)

# %% [markdown]
# ### II.a: Schedule of direct IC trains from Geneva to Lausanne

# %% [markdown]
# select one typical working day.

# %% language="spark"
# sbb_ic = sbb.filter(sbb.betriebstag=='05.11.2018')\
#             .filter(sbb.verkehrsmittel_text=='IC')\
#             .filter(functions.lower(sbb.zusatzfahrt_tf)!='true')\
#             .filter(functions.lower(sbb.faellt_aus_tf)!='true')\
#             .filter(functions.lower(sbb.durchfahrt_tf)!='true')

# %% [markdown]
# schedule of IC trains departing from Geneva

# %% language="spark"
# gen_dep = sbb_ic.filter(sbb_ic.haltestellen_name=='Genève')\
#                 .select(sbb_ic.linien_id, 
#                         functions.to_timestamp(sbb_ic.abfahrtszeit, 'dd.MM.yyyy HH:mm').alias('departure'))

# %% [markdown]
# schedule of IC trains arriving at Lausanne

# %% language="spark"
# lau_arr = sbb_ic.filter(sbb_ic.haltestellen_name=='Lausanne')\
#                 .select(sbb_ic.linien_id, 
#                         functions.to_timestamp(sbb_ic.ankunftszeit, 'dd.MM.yyyy HH:mm').alias('arrival'))

# %% [markdown]
# join two dataframes and perform query

# %% language="spark"
# hour_minute = functions.udf(lambda x: x.strftime('%H:%M'))

# %% language="spark"
# gen_dep.join(lau_arr, gen_dep.linien_id==lau_arr.linien_id, 'inner')\
#        .filter('departure < arrival')\
#        .select(gen_dep.linien_id, 
#                hour_minute(gen_dep.departure).alias('departure'), 
#                hour_minute(lau_arr.arrival).alias('arrival'))\
#        .orderBy('linien_id')\
#        .show()

# %% [markdown]
# ### II.b: Plot arrival delay distribution (in minute) of train IC 733 at Lausanne

# %% magic_args="-o delay" language="spark"
# delay = sbb.filter(sbb.haltestellen_name=='Lausanne')\
#            .filter(sbb.verkehrsmittel_text=='IC')\
#            .filter(sbb.linien_id=='733')\
#            .filter(sbb.an_prognose_status=='REAL')\
#            .select(functions.to_timestamp(sbb.ankunftszeit, 'dd.MM.yyyy HH:mm').alias('expected'),
#                    functions.to_timestamp(sbb.an_prognose, 'dd.MM.yyyy HH:mm:ss').alias('actual'))\
#            .select(functions.when(functions.col('actual') > functions.col('expected'),
#                                   functions.floor((functions.col('actual').cast('long') - functions.col('expected').cast('long')) / 60))
#                             .otherwise(0).alias('delay'))\
#            .groupBy('delay').count().orderBy('delay')

# %%
plt.rcParams['figure.figsize'] = (10, 6)
delay.plot.bar('delay', 'count', legend='', logy=True)
plt.xlabel('Delay/min')
plt.title('Arrival delay distribution of IC 733 at Lausanne');

# %% [markdown]
# ## PART III: Partitioning

# %% [markdown]
# In last week's lecture, we learnt about Spark optimization with partitioning. In this part, here are some examples about how to partition data in PySpark.
#
# First, let's create a sample dataframe from the twitter hashtag dataframe above and check how many partitions it has by default.

# %% language="spark"
# sample_df = twitter_hashtag.sampleBy('lang', {'fr': 0.005, 'en': 0.0004}).select(functions.col('hashtag'), functions.col('lang')).cache()
# sample_df.rdd.getNumPartitions()

# %% [markdown]
# Spark function `spark_partition_id()` enables us to know which partition the row belongs to.

# %% language="spark"
# sample_df.withColumn('pid', functions.spark_partition_id().alias("pid")).show()

# %% [markdown]
# Alternatively, you can use `glom()` to have an RDD created by coalescing all elements within each partition into a list.

# %% language="spark"
# sample_df.rdd.glom().collect()

# %% [markdown]
# **`coalesce`** method reduces the number of partitions in a DataFrame.

# %% language="spark"
# sample_df_5 = sample_df.coalesce(5)
# sample_df_5.rdd.getNumPartitions()

# %% language="spark"
# sample_df_5.withColumn('pid', functions.spark_partition_id().alias("pid")).show()

# %% [markdown]
# If you try to increase the number of partitions with `coalesce`, it won't work.

# %% language="spark"
# sample_df.coalesce(50).rdd.getNumPartitions()

# %% [markdown]
# **`repartition`** method can be used to either increase or decrease the number of partitions in a DataFrame.

# %% language="spark"
# # decrease
# sample_df.repartition(2).rdd.getNumPartitions()

# %% language="spark"
# # increase
# sample_df.repartition(50).rdd.getNumPartitions()

# %% [markdown]
# __Difference between `coalesce` and `repartition`__: 
# - `repartition` does a full shuffle of the data and tries to create equal sized partitions of data
# - `coalesce` combines existing partitions to avoid a full shuffle, e.g. if you go from 1000 partitions to 100 partitions, there will not be a shuffle, instead each of the 100 new partitions will claim 10 of the current partitions. If a larger number of partitions is requested, it will stay at the current number of partitions.

# %% [markdown]
# **Repartition by column**
#
# When partitioning by a column, Spark will create a minimum of 200 partitions by default. The following example will have only 2 partitions with data (one for `en` and another of `fr`) and 198 empty partitions.
#
# Note how the dataframe looks now, compared to the two dataframes above.

# %% language="spark"
# sample_df_lang = sample_df.repartition('lang')
# sample_df_lang.rdd.getNumPartitions()

# %% language="spark"
# sample_df_lang.withColumn('pid', functions.spark_partition_id().alias("pid")).show()

# %%
