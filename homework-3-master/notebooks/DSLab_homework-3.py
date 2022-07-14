# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DSLab Homework3 - Uncovering World Events using Twitter Hashtags
#
# ## ... and learning about Spark `DataFrames` along the way
#
# In this notebook, we will use temporal information about Twitter hashtags to discover trending topics and potentially uncover world events as they occurred. 
#
# ## Hand-in Instructions:
#
# - __Due: 26.04.2022 23:59:59 CET__
# - your project must be private
# - `git push` your final verion to the master branch of your group's Renku repository before the due date
# - check if `Dockerfile`, `environment.yml` and `requirements.txt` are properly written
# - add necessary comments and discussion to make your codes readable

# %% [markdown]
# ## Hashtags
#
# The idea here is that when an event is happening and people are having a conversation about it on Twitter, a set of uniform hashtags that represent the event spontaneously evolves. Twitter users then use those hashtags to communicate with one another. Some hashtags, like `#RT` for "retweet" or just `#retweet` are used frequently and don't tell us much about what is going on. But a sudden appearance of a hashtag like `#oscars` probably indicates that the oscars are underway. For a particularly cool example of this type of analysis, check out [this blog post about earthquake detection using Twitter data](https://blog.twitter.com/official/en_us/a/2015/usgs-twitter-data-earthquake-detection.html) (although they search the text and not necessarily hashtags).

# %% [markdown]
# ## Initialize the environment

# %%
# %load_ext sparkmagic.magics

# %%
import os
from IPython import get_ipython
username = os.environ['RENKU_USERNAME']
server = "http://iccluster029.iccluster.epfl.ch:8998"

# set the application name as "<your_gaspar_id>-homework3"
get_ipython().run_cell_magic(
    'spark',
    line='config', 
    cell="""{{ "name": "{0}-homework3", "executorMemory": "6G", "executorCores": 4, "numExecutors": 15, "driverMemory": "4G"}}""".format(username)
)
#cell="""{{ "name": "{0}-homework3", "executorMemory": "4G", "executorCores": 4, "numExecutors": 10, "driverMemory": "4G"}}""".format(username)

# %% [markdown]
# Send `username` to Saprk kernel, which will frist start the Spark application if there is no active session.

# %%
get_ipython().run_line_magic(
    "spark", "add -s {0}-homework3 -l python -u {1} -k".format(username, server)
)

# %% language="spark"
# print('We are using Spark %s' % spark.version)

# %% language="spark"
# import pyspark
# pyspark.__version__

# %%
# %%spark?

# %% [markdown]
# ## PART I: Set up (5 points)
#
# The twitter stream data is downloaded from [Archive Team: The Twitter Stream Grab](https://archive.org/details/twitterstream), which is a collection of a random sample of all tweets. We have parsed the stream data and prepared the twitter hashtag data of __2020__, a very special and different year in many ways. Let's see if we can see any trends about all these events of 2020 in the Twitter data. 
#
# <div style="font-size: 100%" class="alert alert-block alert-danger">
# <b>Disclaimer</b>
# <br>
# This dataset contains unfiltered data from Twitter. As such, you may be exposed to tweets/hashtags containing vulgarities, references to sexual acts, drug usage, etc.
# </div>

# %% [markdown]
# ### a) Load data - 1/10
#
# Load the **orc** data from `/data/twitter/orc/hashtags/year=2020` into a Spark dataframe using the appropriate `SparkSession` method. 
#
# Look at the first few rows of the dataset - note the timestamp and its units!

# %% language="spark"
# # We load the data from the part_orc folder instead of the orc folder (as explained on Slack)
# df = spark.read.orc("/data/twitter/part_orc/hashtags/year=2020")
# df.printSchema()

# %% language="spark"
# df.show(n=5, truncate=False, vertical=False)

# %% [markdown]
# <div style="font-size: 100%" class="alert alert-block alert-info">
#     <b>Cluster Usage:</b> As there are many of you working with the cluster, we encourage you to
#     <ul>
#         <li>prototype your queries on small data samples before running them on whole datasets</li>
#         <li>save your intermediate results in your own directory at hdfs <b>"/user/&lt;your-gaspar-id&gt;/"</b></li>
#     </ul>
# </div>
#
# For example:
#
# ```python
#     # create a subset of original dataset
#     df_sample = df.sample(0.01)
#     
#     # save as orc
#     df_sample.write.orc('/user/%s/sample.orc' % username, mode='overwrite')
#
# ```

# %% [markdown]
# ### b) Functions - 2/10

# %% language="spark"
# import pyspark.sql.functions as F

# %% [markdown]
# __User-defined functions__
#
# A neat trick of spark dataframes is that you can essentially use something very much like an RDD `map` method but without switching to the RDD. If you are familiar with database languages, this works very much like e.g. a user-defined function in SQL. 
#
# So, for example, if we wanted to make a user-defined python function that returns the hashtags in lowercase, we could do something like this:

# %% language="spark"
# @F.udf
# def lowercase(text):
#     """Convert text to lowercase"""
#     return text.lower()

# %% [markdown]
# The `@F.udf` is a "decorator" -- this is really handy python syntactic sugar and in this case is equivalent to:
#
# ```python
# def lowercase(text):
#     return text.lower()
#     
# lowercase = F.udf(lowercase)
# ```
#
# It basically takes our function and adds to its functionality. In this case, it registers our function as a pyspark dataframe user-defined function (UDF).
#
# Using these UDFs is very straightforward and analogous to other Spark dataframe operations. For example:

# %% language="spark"
# df.select(lowercase(df.hashtag)).show(n=5)

# %% [markdown]
# __Built-in functions__
#
# Using a framework like Spark is all about understanding the ins and outs of how it functions and knowing what it offers. One of the cool things about the dataframe API is that many functions are already defined for you (turning strings into lowercase being one of them). Find the [Spark python API documentation](https://spark.apache.org/docs/2.3.2/api/python/index.html). Look for the `sql` section and find the listing of `sql.functions`. Repeat the above (turning hashtags into lowercase) but use the built-in function.

# %% language="spark"
# df.select(F.lower(df.hashtag)).show(n=5)

# %% [markdown]
# We'll work with a combination of these built-in functions and user-defined functions for the remainder of this homework. 
#
# Note that the functions can be combined. Consider the following dataframe and its transformation:

# %% language="spark"
# from pyspark.sql import Row
#
# # create a sample dataframe with one column "degrees" going from 0 to 180
# test_df = spark.createDataFrame(spark.sparkContext.range(180).map(lambda x: Row(degrees=x)), ['degrees'])
#
# # define a function "sin_rad" that first converts degrees to radians and then takes the sine using built-in functions
# sin_rad = F.sin(F.radians(test_df.degrees))
#
# # show the result
# test_df.select(sin_rad).show()

# %% [markdown]
# ### c) Tweets in english - 2/10
#
# - Create `english_df` with only english-language tweets. 
# - Turn hashtags into lowercase.
# - Convert the timestamp to a more readable format and name the new column as `date`.
# - Sort the table in chronological order. 
#
# Your `english_df` should look something like this:
#
# ```
# +-----------+----+-----------+-------------------+
# |timestamp_s|lang|    hashtag|               date|
# +-----------+----+-----------+-------------------+
# | 1577862000|  en| spurfamily|2020-01-01 08:00:00|
# | 1577862000|  en|newyear2020|2020-01-01 08:00:00|
# | 1577862000|  en|     master|2020-01-01 08:00:00|
# | 1577862000|  en|  spurrific|2020-01-01 08:00:00|
# | 1577862000|  en|     master|2020-01-01 08:00:00|
# +-----------+----+-----------+-------------------+
# ```
#
# __Note:__ 
# - The hashtags may not be in english.
# - [pyspark.sql.functions](https://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html#module-pyspark.sql.functions)

# %% language="spark"
# df.select('timestamp_s').show(n=5)

# %% language="spark"
#
# english_df = df.filter(df.lang=='en')\
#                 .select('timestamp_s','lang',F.lower(df.hashtag).alias('hashtag'),F.from_unixtime('timestamp_s', format='yyyy-MM-dd HH:mm:ss').alias('date'))\
#                 .sort('timestamp_s',ascending=True)
#                                              

# %% language="spark"
# english_df.show(n=5)

# %% [markdown]
# ## PART II: Twitter hashtag trends (30 points)
#
# In this section we will try to do a slightly more complicated analysis of the tweets. Our goal is to get an idea of tweet frequency as a function of time for certain hashtags. 
#
# Have a look [here](http://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html#module-pyspark.sql.functions) to see the whole list of custom dataframe functions - you will need to use them to complete the next set of TODO items.

# %% [markdown]
# ### a) Top hashtags - 1/30
#
# We used `groupBy` already in the previous notebooks, but here we will take more advantage of its features. 
#
# One important thing to note is that unlike other RDD or DataFrame transformations, the `groupBy` does not return another DataFrame, but a `GroupedData` object instead, with its own methods. These methods allow you to do various transformations and aggregations on the data of the grouped rows. 
#
# Conceptually the procedure is a lot like this:
#
# ![groupby](https://i.stack.imgur.com/sgCn1.jpg)
#
# The column that is used for the `groupBy` is the `key` - once we have the values of a particular key all together, we can use various aggregation functions on them to generate a transformed dataset. In this example, the aggregation function is a simple `sum`. In the simple procedure below, the `key` will be the hashtag.
#
#
# Use `groupBy`, calculate the top 5 most common hashtags in the whole english-language dataset.
#
# This should be your result:
#
# ```
# +-----------------+-------+
# |          hashtag|  count|
# +-----------------+-------+
# |              bts|1200196|
# |          endsars|1019280|
# |          covid19| 717238|
# |            방탄소년단| 488160|
# |sarkaruvaaripaata| 480124|
# +-----------------+-------+
# ```

# %% language="spark"
# top_5_hashtag =(english_df.groupBy('hashtag')\
#             .count()\
#             .sort('count',ascending=False)\
# )
# top_5_hashtag.show(n=5)

# %% [markdown]
# ### b) Daily hashtags - 2/50
#
# Now, let's see how we can start to organize the tweets by their timestamps. Remember, our goal is to uncover trending topics on a timescale of a few days. A much needed column then is simply `day`. Spark provides us with some handy built-in dataframe functions that are made for transforming date and time fields.
#
# - Create a dataframe called `daily_hashtag` that includes the columns `month`, `week`, `day` and `hashtag`. 
# - Use the `english_df` you made above to start, and make sure you find the appropriate spark dataframe functions to make your life easier. For example, to convert the date string into day-of-year, you can use the built-in [dayofyear](http://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html#pyspark.sql.functions.dayofyear) function. 
# - For the simplicity of following analysis, filter only tweets of 2020.
# - Show the result.
#
# Try to match this view:
#
# ```
# +-----+----+---+-----------+
# |month|week|day|    hashtag|
# +-----+----+---+-----------+
# |    1|   1|  1| spurfamily|
# |    1|   1|  1|newyear2020|
# |    1|   1|  1|     master|
# |    1|   1|  1|  spurrific|
# |    1|   1|  1|     master|
# +-----+----+---+-----------+
# ```

# %% language="spark"
# daily_hashtag = english_df.filter(F.year(english_df.date)==2020)\
#                         .sort('timestamp_s')\
#                         .select(F.month(english_df.date).alias('month'),\
#                                 F.weekofyear(english_df.date).alias('week'),\
#                                 F.dayofyear(english_df.date).alias('day'),\
#                                 'hashtag')                                        
# daily_hashtag.show(n=5)
#

# %% [markdown]
# ### c) Daily counts - 2/50
#
# Now we want to calculate the number of times a hashtag is used per day based on the dataframe `daily_hashtag`. Sort in descending order of daily counts and show the result. Call the resulting dataframe `day_counts`.
#
# Your output should look like this:
#
# ```
# +---+----------------------+----+------+
# |day|hashtag               |week|count |
# +---+----------------------+----+------+
# |229|pawankalyanbirthdaycdp|33  |202241|
# |222|hbdmaheshbabu         |32  |195718|
# |228|pawankalyanbirthdaycdp|33  |152037|
# |357|100freeiphone12       |52  |122068|
# |221|hbdmaheshbabu         |32  |120401|
# +---+----------------------+----+------+
# ```
#
# <div class="alert alert-info">
# <p>Make sure you use <b>cache()</b> when you create <b>day_counts</b> because we will need it in the steps that follow!</p>
# </div>

# %% language="spark"
# day_counts = daily_hashtag.groupBy('day','hashtag','week')\
#                         .agg(F.count('hashtag').alias('count'))\
#                         .sort('count',ascending=False)\
#                         .cache()

# %% language="spark"
# day_counts.show(n=5, truncate=False)

# %% [markdown]
# ### d) Weekly average - 2/50
#
# To get an idea of which hashtags stay popular for several days, calculate the average number of daily occurences for each week. Sort in descending order and show the top 20.
#
# __Note:__
# - Use the `week` column we created above.
# - Calculate the weekly average using `F.mean(...)`.

# %% language="spark"
#
# day_counts.groupby('hashtag','week')\
#         .agg(F.mean('count').alias('mean'))\
#         .sort('mean', ascending=False)
# day_counts.show(n=20)

# %% [markdown]
# ### e) Ranking - 3/20
#
# Window functions are another awesome feature of dataframes. They allow users to accomplish complex tasks using very concise and simple code. 
#
# Above we computed just the hashtag that had the most occurrences on *any* day. Now lets say we want to know the top tweets for *each* day.  
#
# This is a non-trivial thing to compute and requires "windowing" our data. I recommend reading this [window functions article](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html) to get acquainted with the idea. You can think of a window function as a fine-grained and more flexible `groupBy`. 
#
# There are two things we need to define to use window functions:
#
# 1. the "window" to use, based on which columns (partitioning) and how the rows should be ordered 
# 2. the computation to carry out for each windowed group, e.g. a max, an average etc.
#
# Lets see how this works by example. We will define a window function, `daily_window` that will partition data based on the `day` column. Within each window, the rows will be ordered by the daily hashtag count that we computed above. Finally, we will use the rank function **over** this window to give us the ranking of top tweets. 
#
# In the end, this is a fairly complicated operation achieved in just a few lines of code! (can you think of how to do this with an RDD??)

# %% language="spark"
# from pyspark.sql import Window

# %% [markdown]
# First, we specify the window function and the ordering:

# %% language="spark"
# daily_window = Window.partitionBy('day').orderBy(F.desc('count'))

# %% [markdown]
# The above window function says that we should window the data on the `day` column and order it by count. 
#
# Now we need to define what we want to compute on the windowed data. We will start by just calculating the daily ranking of hashtags, so we can use the helpful built-in `F.rank()` and sort:

# %% language="spark"
# daily_rank = F.rank() \
#               .over(daily_window) \
#               .alias('rank')

# %% [markdown]
# Now compute the top five hashtags for each day in our data:

# %% language="spark"
# day_counts_rank = day_counts.withColumn('rank', daily_rank)
# top5_each_day = day_counts_rank.filter(day_counts_rank.rank<=5).sort('day','week','rank')
# top5_each_day.show(n=20)

# %% [markdown]
# ### f) Rolling sum - 5/30
#
# With window functions, you can also calculate the statistics of a rolling window. 
#
# In this question, construct a 7-day rolling window (including the day and 6 days before) to calculate the rolling sum of the daily occurences for each hashtag.
#
# Your results should be like:
# - For the hashtag `covid19`:
#
# ```
# +---+----+-----+-------+-----------+
# |day|week|count|hashtag|rolling_sum|
# +---+----+-----+-------+-----------+
# | 42|   7|   85|covid19|         85|
# | 43|   7|   94|covid19|        179|
# | 45|   7|  192|covid19|        371|
# | 46|   7|   97|covid19|        468|
# | 47|   7|  168|covid19|        636|
# | 48|   8|  317|covid19|        953|
# | 49|   8|  116|covid19|        984|
# | 51|   8|  234|covid19|       1124|
# | 52|   8|  197|covid19|       1129|
# | 53|   8|  369|covid19|       1401|
# +---+----+-----+-------+-----------+
# ```
#
# - For the hashtag `bts`:
#
# ```
# +---+----+-----+-------+-----------+
# |day|week|count|hashtag|rolling_sum|
# +---+----+-----+-------+-----------+
# |  1|   1| 2522|    bts|       2522|
# |  2|   1| 1341|    bts|       3863|
# |  3|   1|  471|    bts|       4334|
# |  4|   1|  763|    bts|       5097|
# |  5|   1| 2144|    bts|       7241|
# |  6|   2| 1394|    bts|       8635|
# |  7|   2| 1673|    bts|      10308|
# |  8|   2| 5694|    bts|      13480|
# |  9|   2| 5942|    bts|      18081|
# | 10|   2| 5392|    bts|      23002|
# +---+----+-----+-------+-----------+
# ```

# %% language="spark"
# # define the window to be the six previous day and the current day for each hashtag
# rolling_window = Window.partitionBy('hashtag').orderBy(F.asc('day')).rangeBetween(-6,0)
#
# # Define the custom rolling function to be a sum of the count voer the days
# day_counts_sum = day_counts.withColumn('rolling_sum', F.sum('count')\
#                                        .over(rolling_window))
#
# # Put the result in the required format
# rs_counts = day_counts_sum.select('day','week','count','hashtag','rolling_sum')\
#                             .sort('day',ascending=True)

# %% language="spark"
# rs_counts.filter('hashtag == "covid19"').show(n=10)

# %% language="spark"
# rs_counts.filter('hashtag == "bts"').show(n=10)

# %% [markdown]
# We can see that we get the expected resutls for the hashtags `bts` and `covid19`

# %% [markdown]
# ### g) DIY - 15/20
#
# Use window functions (or other techniques!) to produce lists of top few trending tweets for each week. What's a __"trending"__ tweet? Something that seems to be __suddenly growing very rapidly in popularity__. 
#
# You should be able to identify, for example, Oscars-related hashtags in week 7 when [the 92nd Academy Awards ceremony took place](https://www.oscars.org/oscars/ceremonies/2020), COVID-related hashtags in week 11 when [WHO declared COVID-19 a pandemic](https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020), and other events like the movement of Black Life Matters in late May, the United States presidential elections, the 2020 American Music Awards, etc.
#
# The final listing should be clear and concise and the flow of your analysis should be easy to follow. If you make an implementation that is not immediately obvious, make sure you provide comments either in markdown cells or in comments in the code itself.

# %% [markdown]
# **Answer:**
# As described above, a trending tweets is a tweet that has suddenly grows very rapidely in popularity, i.e. a tweet for which the count between the previous and the current is large.
#
# The general idea we thought of is the following:
# 1. Compute the weekly count for every hashtag.
# 2. Compute the difference between the current count and the previous one for each week.
# 3. Rank the difference and take the 10 highest.
#
# After that, we will look in more details at the results we get and assess if this methodology does the job or not. For simplicity, as we have only data for the 2020 year, we decided to exclude the first week as we have no idea if the hashtags of week 1 were or weren't trending the last week of 2019. 

# %% language="spark"
# # Part 1: weekly count of hashtag apparition
# weekly_count = day_counts.groupby('hashtag','week')\
#                         .agg(F.sum('count').alias('count'))
#
# # Part 2: add new column with the count lagged by one week (add the count of the previous week as a new column for each week)
# rolling_window = Window.partitionBy('week')\
#                         .orderBy(F.asc('week'))
#
# weekly_count = weekly_count.withColumn("prev_count", \
#                                        F.lag('count', count=1, default=0).over(rolling_window))
#
# # Compute the difference between a week and the previous week and adds it as a new column
# weekly_count = weekly_count.withColumn("diff", F.col('count')-F.col('prev_count'))
#
# # Create a partition over the week sorted by decreasing difference and compute the rank over the window
# weekly_window_count = Window.partitionBy('week').orderBy(F.desc('diff'))
# weekly_rank = F.rank().over(weekly_window_count).alias('rank')
#
# # Add the rank as a new column for each row 
# weekly_count = weekly_count.withColumn('rank', weekly_rank)
#
# # Filter out the hashtag with rank greater than 5 and filter out week 1
# top_5_week = weekly_count.filter((weekly_count.rank<=5) & (weekly_count.week>1))
#
# top_5_week.show()

# %% language="spark"
# weekly_count.filter(weekly_count.week==11).show(n=10)

# %% language="spark"
# weekly_count.filter(weekly_count.week==7).show(n=10)

# %% language="spark"
# weekly_count.filter(weekly_count.week==22).show(n=10)

# %% [markdown]
# From this, we can see that in week 11, we are able to spot the oscar related hashtags. Also, we can see the rapdingly growing unsafety about `covid19` in week 11 (corresponding to the end of March) introduces the hashtags `coronavirus` and `covid19`. We can see that our method is indeed able to spot the most trending hashtag in the sense defined above. 

# %% [markdown]
# ## PART III: Hashtag clustering (25 points)

# %% [markdown]
# ### a) Feature vector - 3/25
#
# - Create a dataframe `daily_hashtag_matrix` that consists of hashtags as rows and daily counts as columns (hint: use `groupBy` and methods of `GroupedData`). Each row of the matrix represents the time series of daily counts of one hashtag. Cache the result.
#
# - Create the feature vector which consists of daily counts using the [`VectorAssembler`](https://spark.apache.org/docs/2.3.2/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler) from the Spark ML library. Cache the result.

# %% language="spark"
# # Set count value to 0 instead of null (if there were no hashtag for this day)
# daily_hashtag_matrix = english_df.select('hashtag', F.dayofyear(english_df.date).alias('day')).withColumn('count', F.lit(1))\
#                         .groupby('hashtag')\
#                         .pivot('day')\
#                         .sum('count')\
#                         .fillna(0).cache()

# %% [markdown]
# We observed there were approximately 10 days missing. For the indexing of the rows to correspond to the days, we fill these missing days with 0s.

# %% language="spark"
# from pyspark.sql.functions import lit
#
# # There are some missing days, set the data for these days to 0
# missing_days = set([str(i) for i in range(1, 367)] + ['hashtag']) - set(daily_hashtag_matrix.columns)
#
# # So fill them with 0s for all hashtags to get a correct plot below
# for md in missing_days:
#     daily_hashtag_matrix = daily_hashtag_matrix.withColumn(md, lit(0))
#     
# daily_hashtag_matrix_sorted = daily_hashtag_matrix.select(['hashtag'] + sorted(list(set(daily_hashtag_matrix.columns) - set(['hashtag'])), key=lambda x: int(x)))

# %% language="spark"
# from pyspark.ml.feature import VectorAssembler
#
# # Build the features vectors for each day
# columns_features = daily_hashtag_matrix_sorted.drop('hashtag').columns
#
# vector_assembler = VectorAssembler(inputCols=columns_features, outputCol='features')
# daily_vector = vector_assembler.transform(daily_hashtag_matrix_sorted).cache()

# %% [markdown]
# ### b) Visualization - 2/25
#
# Visualize the time sereis you just created. 
#
# - Select a few interesting hashtags you identified above. `isin` method of DataFrame columns might be useful.
# - Retrieve the subset DataFrame using sparkmagic
# - Plot the time series for the chosen hashtags with matplotlib.

# %% [markdown]
# ---
# We decided to consider the following hashtags : ```['oscars', 'coronavirus', 'blacklivesmatter', 'election2020', 'mamavote']```.

# %% magic_args="-o df_plot" language="spark"
# from pyspark.sql.types import *
#
# # Convert Vector to a list
# toArray = F.udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
#
# # Filter the hastags
# hashtags = ['oscars', 'coronavirus', 'blacklivesmatter', 'election2020', 'amas']
# tmp = daily_vector.filter(daily_vector.hashtag.isin(hashtags))
#
# df_plot = tmp.select('hashtag', toArray(tmp.features).alias('features'))

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (14,10)

# Plot the daily counts for these hashtags
for i, row in df_plot.iterrows():
    plt.plot(range(len(row['features'])), row['features'], label=row['hashtag'])
    

plt.title('Daily counts of selected hashtags')
plt.xlabel('Days')
plt.ylabel('Count')
plt.legend(loc=2)
plt.show()

# %% [markdown]
# We can observe different kind of behaviors in the plot above :
# - Some series have a very precise and large spike whereas other are more spread out. For example, the Oscars are very concentrated around the data of the event compared to the covid19 which much more spread out over the year.
# - Some series have a very clear starting point whereas others not. For example, the blacklivesmatter has a clear starting point when the event occured, followed by a spike and then decreasing gently. Others such as the covid19 do not have such clear starting points.
# - We can also observe a difference in the scale of the daily counts. For example, the election2020 and amas (American Music Awards) happened in the same period but have very different scales.

# %% [markdown]
# ### c) KMeans clustering - 20/25
#
# Use KMeans to cluster hashtags based on the daily count timeseries you created above. Train the model and calculate the cluster membership for all hashtags. Again, be creative and see if you can get meaningful hashtag groupings. 
#
# Validate your results by showing certain clusters, for example, those including some interesting hashtags you identified above. Do they make sense?
#
# Make sure you document each step of your process well so that your final notebook is easy to understand even if the result is not optimal or complete. 
#
# __Note:__ 
# - Additional data cleaning, feature engineering, deminsion reduction, etc. might be necessary to get meaningful results from the model. 
# - For available methods, check [pyspark.sql.functions documentation](https://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html#module-pyspark.sql.functions), [Spark MLlib Guide](https://spark.apache.org/docs/2.3.2/ml-guide.html) and [pyspark.ml documentation](https://spark.apache.org/docs/2.3.2/api/python/pyspark.ml.html).

# %% [markdown]
# ---
# ### Using basic daily counts as features

# %% language="spark"
# from pyspark.ml.clustering import KMeans
# from pyspark.ml.feature import StandardScaler
# from pyspark.ml.evaluation import ClusteringEvaluator
#
# def kmeans(df, features_col, k, seed=0):
#     """
#     Run the KMeans algorithms on the given features
#     :param df : dataframe containing the features
#     :param features_col : name of the column containing the features
#     :param k : number of clusters
#     :param seed : seed for the random initialization
#     """
#     # Instantiate model
#     kmeans = KMeans(k=k, seed=seed, featuresCol=features_col)
#
#     # "Train" the model
#     model = kmeans.fit(df)
#
#     # Make the "predictions", i.e. assign the cluster to each hashtag
#     df_transformed = model.transform(df)
#     
#     return df_transformed, model
#
# def show_cluster(df, cluster_id, n=10):
#     """
#     Show some of the hashtags present in the given cluster
#     :param df : dataframe containing the hashtag and their cluster id
#     :param cluster_id : id of the cluster we want to show
#     :param n : number of rows to show
#     """
#     return df.filter(df.prediction == cluster_id).select('hashtag').show(n=n)
#
# def silouhette_scores(df, features_col, ks=[5, 8, 10, 12, 15], seed=0):
#     """
#     Compute the sihoulette score of KMeans for different
#     values of k.
#     :param df : dataframe containing the features
#     :param features_col : name of the column containing the features
#     :param ks : values of k to try
#     """
#     scores = []
#     for k in ks:
#         df_kmeans, model = kmeans(df, features_col, k=k, seed=seed)
#         
#         # Evaluate clustering by computing Silhouette score
#         evaluator = ClusteringEvaluator(featuresCol=features_col)
#         scores.append((k, evaluator.evaluate(df_kmeans)))
#         
#     return scores

# %% language="spark"
#
# # Run KMeans
# df_kmeans, model = kmeans(daily_vector, 'features', k=5)
#
# model.summary.clusterSizes

# %% [markdown]
# We directly see that this clustering is useless since almost all the points are in a single cluster, i.e. this gives virtually no insight into the data. So we need to craft better features.

# %% [markdown]
# ---
# ### Data cleaning and preprocessing
#
# In this step, we aim at filtering out hashtags that make it harder to obtain meaningful (i.e. related to current events in 2020) clusters. In particular, we make the following observation :
#
# - Hashtags that have very small variance in the daily counts are hashtags that people use every day. Therefore they do not convey information about current events. We filter them out as follows :
#     - Compute the standard deviation (std) of the daily counts time series and filter out the one having less than 300, based on the std of few interesting hashtags such as ```['oscars', 'coronavirus', 'blacklivesmatter', 'election2020', 'mamavote']```.

# %% language="spark"
# import numpy as np
#
# # Compute the std of the daily count for each hashtag
# std_counts = F.udf(lambda vector: float(np.std(vector.toArray().tolist())), DoubleType())
#
# daily_vector = daily_vector.withColumn('std_daily_count', std_counts(daily_vector.features))

# %% language="spark"
#
# # Print some of the stds for interesting hashtags
# tmp = daily_vector.filter(daily_vector.hashtag.isin(hashtags))
# tmp.select('hashtag', 'std_daily_count').show(n=5)

# %% [markdown]
# Here we set the std threshold to 300 to allow all the above selected hashtags to pass the filter. We also noted that this filtering is quite selective, so we set it lower than the smallest std above (approx. 460).

# %% language="spark"
#
# # Set the threshold for the minimal std
# std_threshold = 300
#
# # Filter out the hashtag with a lower std
# daily_vector_threshold = daily_vector.filter(daily_vector.std_daily_count > std_threshold)

# %% [markdown]
# In the following steps we introduce some other features that we will combine in the end.

# %% [markdown]
# ---
# ### Features based on the dimensionality reduction of the daily counts
#
# KMeans is also well known for suffering of the curse of dimensionality. Therefore we need to reduce the number of dimensions of the input. Here the aim is to keep the information of the daily counts intact (as possible) but to drastically reduce the number of dimensions. For this we use the PCA algorithms on the entire daily counts time series of each hashtag.
#
#
# There might be large differences in the total counts. For example, we see from the plot in part 3.b), that the election2020 has a very low counts compared to mamavote (American Music Awards). However, we would like to cluster them together as they happen approximately at the same time (from the plot above). So we normalize the vector of daily counts such that each time series has a unit L2-norm. This also helps for the PCA.

# %% language="spark"
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.feature import Normalizer
# from pyspark.ml.feature import PCA
# from pyspark.ml.linalg import Vectors
#
# # Setup the pipeline
# # Normalize each vector (time series) to have unit 2-norm
# normalizer = Normalizer(inputCol="features", outputCol="normalized_features")
#
# # Project on first 5 principal components
# pca = PCA(k=5, inputCol="normalized_features", outputCol="pca_features_daily_count")
#
# pipeline = Pipeline(
#     stages=[
#         normalizer,
#         pca
#     ]
# )
#
# df_transformed = (pipeline.fit(daily_vector_threshold).transform(daily_vector_threshold)).cache()

# %% magic_args="-o df_scores" language="spark"
# from pyspark.sql import Row
# from operator import itemgetter
#
# # Compute the silouhette scores for different values of k
# ks = list(range(3, 20))
# scores = silouhette_scores(df_transformed, 'pca_features_daily_count', ks=ks)
#
# # Get the best k
# best_k = max(scores, key=itemgetter(1))[0]
#
# print('Best k : {0}'.format(best_k))
#
# # Get back the results locally, have to create a pypspark df...
# df_scores = map(lambda x: Row(*x), scores)
# df_scores = spark.createDataFrame(df_scores, ['k', 'silouhette'])

# %%
# Plot the scores for each value of k
plt.plot(df_scores['k'], df_scores['silouhette'], marker='o')

plt.title('Silouhette scores for different $k$')
plt.xlabel('k')
plt.ylabel('Silouhette score')
plt.show()

# %% [markdown]
# The above figure shows the different silouette score for our K-means results (the higher the better). In a "sequential" setting, i.e. where all the data are stored in a single machine the results of the Kmeans is deterministic when the initialization is fixed. However, since we are in a distributed setting here, it appears that the results will be somewhat random (even if the seed is fixed) depending on the data locality, [as explained here](https://stackoverflow.com/questions/69731313/performing-pca-in-pyspark-returns-different-results-each-run). The following cells correspond to an analysis of a run where we get a $best\_k= 6$ and where we looked in details at the content of each cluster. A possible solution to fix the issue may be to specify the data partitioning in Spark but we haven't explore this solution in details.

# %% language="spark"
#
# # Compute the Kmeans with the best k
# df_all_kmeans, model_all = kmeans(df_transformed, 'pca_features_daily_count', k=best_k)
#
# # Show the cluster sizes
# model_all.summary.clusterSizes

# %% language="spark"
#
# # Print to which cluster each of our hashtags belong to
# df_all_kmeans.filter(df_all_kmeans.hashtag.isin(hashtags)).select('hashtag', 'prediction').show(n=10)

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 7, n=15)

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 5, n=5)

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 4, n=5)

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 3, n=5)

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 2, n=5)

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 1, n=5)

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 0, n=5)

# %% [markdown]
# #### Let's try to visualize the result of the clustering

# %% magic_args="-o df_cluster_plot" language="spark"
#
# input_cols = [str(x) for x in range(1, 367)]   
# # Save the greatest daily count for each hashtag for visualization purposes
# df_cluster_plot = df_all_kmeans.withColumn('greatest', F.greatest(*input_cols))

# %%
# Define a map of visually distinct colors for clusters (thanks to https://sashamaps.net/docs/resources/20-colors/)
COLOR_MAP = {
    0: '#e6194B', 1: '#f58231', 2: '#000075',
    3: '#42d4f4', 4: '#3cb44b', 5: '#bfef45',
    6: '#4363d8', 7: '#911eb4', 8: '#f032e6',
    9: '#a9a9a9', 10: '#000000', 11: '#ffe119',
    12: '#469990', 13: '#808000', 14: '#9A6324',
    15: '#800000', 16: '#fabed4', 17: '#ffd8b1',
    18: '#aaffc3', 19: '#dcbeff', 20: '#fffac8'
}


# %%
def plot_clusters(df_kmeans, sample_size=5, min_threshold=None, max_threshold=None):
    '''
    Plot the daily count timeseries for samples derived from each cluster of the dataframe.
    :param df_kmeans : the input dataframe where data is labeled in 'prediction' based on clustering
    :param sample_size : the number of hashtags to sample per cluster
    :param min_threshold: the minimum daily count of hashtags to plot
    :param max_threshold: the maximum daily count of hashtags to plot
    '''
    plt.rcParams["figure.figsize"] = (18,12)
    
    # Get the number of clusters
    nb_clusters = df_kmeans.prediction.unique()
    
    # Days
    input_cols = list(range(1, 367))
    
    # For each cluster derive a sample and plot it
    for i in nb_clusters:
        df_cluster = df_kmeans[df_kmeans.prediction == i]
        
        # Clip the hashtags with daily counts between minimum threshold and maximum threshold if defined
        if (min_threshold):
            df_cluster = df_cluster[(df_cluster.greatest >= min_threshold)]
        if (max_threshold):
            df_cluster = df_cluster[(df_cluster.greatest <= max_threshold)]
        
        # Sample sample_size samples from the cluster
        if (len(df_cluster) >= sample_size):
            df_cluster = df_cluster.sample(n=sample_size, replace=True)
        
        # Plot the daily counts for these hashtags
        for i, row in df_cluster.iterrows():
            plt.plot(input_cols, row[[str(x) for x in input_cols]], c=COLOR_MAP[row['prediction']], label=row['prediction'])
            #display(row[['1', '2']])
    

    plt.title(f'Daily counts of {sample_size} selected hashtags from each cluster for {len(nb_clusters)} clusters')
    plt.xlabel('Days')
    plt.ylabel('Count')
    plt.legend(loc=2)
    plt.show()        

# %%
plot_clusters(df_cluster_plot, sample_size=5, min_threshold=10000, max_threshold=20000)

# %% [markdown]
# ### Clusters description
#
# __Cluster 0:__ multiple peaks every couple of months  
#
# Similar hashtags in the cluster: `[blackswanlive, blackswan, blackswanoutnow]`, `[btsxcorden, btsonagt, curated_by_bts]`, `[blacklivesmatter, blacklivesmatter, black_lives_matter, georgefloyd]` 
#
# This cluster regroups the popular hashtags with the important related events that happenned multiple times during the year, e.g. an artist having a tourney.
#
# __Cluster 1:__ strong peak at the end of the year
#
# Similar hashtags in the cluster: `[endssars, sarsmustend, endsarsprotests, sarsmustendnow, endsarsprotest, endsarsimmediately, endsars, ...]`, `[endpolicebrutality, ...]`, `[endbadgoverance, ...]`
#
# __Cluster 2:__ pretty trandy the whole year, the biggest picks are in the middle of the year
#
# Similar hashtags in the cluster: `[covid19, covid_19]`, `[got7_dye, got7, got7_notbythemoon]`, `[nctdream, nct127]`
#
# __Cluster 3:__ the smallest cluster, less pronounced than the others, one peak at the end of the year
#
# Similar hashtags in the cluster: `[2020mama_stanbot,  mamavote, ...]`
#
# __Cluster 4:__  very popular hashtags, especially in the beginning and in the end of the year
#
# Similar hashtags in the cluster: `[resonance, resonance_pt2]`
#
# __Cluster 5:__ hashtags popular in the beginning of the year
#
# Similar hashtags in the cluster: `[covidー19, coronavirus]`, `[biggboss13, bb13]` 
#
# __Conclusion:__ The resulting clusters regroup the similar hashtags with respect to the time frequency and the related topic.

# %% [markdown]
# With larger k, the clusters can be hard to distinguish from the above visualization. Let's run PCA again on the features to reduce them to 2 dimensions and visualize a scatter plot instead. Let's try this time to plot inside spark and let spark send back the final plot.

# %% language="spark"
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pylab as plt
#
# plt.rcParams['figure.figsize'] = (14,10)
# plt.rcParams['font.size'] = 18
# plt.style.use('fivethirtyeight')
#
# # Define a map of visually distinct colors for clusters (thanks to https://sashamaps.net/docs/resources/20-colors/)
# COLOR_MAP = {
#     0: '#e6194B', 1: '#f58231', 2: '#000075',
#     3: '#42d4f4', 4: '#3cb44b', 5: '#bfef45',
#     6: '#4363d8', 7: '#911eb4', 8: '#f032e6',
#     9: '#a9a9a9', 10: '#000000', 11: '#ffe119',
#     12: '#469990', 13: '#808000', 14: '#9A6324',
#     15: '#800000', 16: '#fabed4', 17: '#ffd8b1',
#     18: '#aaffc3', 19: '#dcbeff', 20: '#fffac8'
# }
#
# def plot_clusters_scatter(df_kmeans, input_col):
#     '''
#     Run PCA on the clustered input dataframe using the provided input column and
#     visualize a scatter plot of the result.
#     
#     :param df_kmeans : the input dataframe where data is labeled in 'prediction' based on clustering
#     :param input_col : the features column on which we're running the PCA to obtain 2 dimensions
#     '''
#     # To vizualize we reduce the dimension of the features to 2 using PCA again
#     pca_plot = PCA(k=2, inputCol=input_col, outputCol="plot_points")
#     df_clusters_scatter = (
#         pca_plot.fit(df_kmeans)
#         .transform(df_kmeans)
#         .select("hashtag", "plot_points", "prediction")
#     )
#     
#     # scatter plot of cluster result of 2500 hashtag samples
#     x = [x['plot_points'][0] for x in df_clusters_scatter.select("plot_points").collect()]
#     y = [y['plot_points'][1] for y in df_clusters_scatter.select("plot_points").collect()]
#     colors = [COLOR_MAP[i['prediction']] for i in df_clusters_scatter.select("prediction").collect()]
#
#     
#     plt.scatter(x, y, c=colors)
#     plt.title('Scatter plot of the clusters after reducing the hashtags daily counts to 2 dimensions using PCA')
#     plt.legend(loc=2)
#     plt.show()    
#     
# plot_clusters_scatter(df_all_kmeans, "pca_features_daily_count")

# %% language="spark"
# %matplot plt

# %% [markdown]
# Below we introduce other features to see if can improve the clustering.

# %% [markdown]
# ---
# ### Features based on emergence and disappearing of the hashtags
#
# We would like to clusterize hashtags of events that happened around the same time. However, as said above, the features about the spectrum of the daily counts do bring any information about this. So we decide to drop them and instead introduce the 3 following features (keeping the projected (PCA) daily count features) :
# - when the daily count first got above 10% of the total count for this hashtag
# - the last time the daily count got below 10% of the total count for this hashtag
# - when the daily count was at its highest

# %% language="spark"
#
# threshold = 0.1
#
# # Compute the first time the daily count got above the threshold
# first_above = F.udf(lambda vector: float(np.argmax((np.array(vector.toArray().tolist()) / float(np.sum(vector.toArray().tolist()))) > threshold)), DoubleType())
#
# # Compute the last time the daily count got below the threshold
# last_below = F.udf(lambda vector: float(365 - np.argmax(((np.array(vector.toArray().tolist()) / float(np.sum(vector.toArray().tolist())))[::-1]) > threshold)), DoubleType())
#
# # Compute when it reaches its maximum
# time_max = F.udf(lambda vector: float(np.argmax(np.array(vector.toArray().tolist()))), DoubleType())
#
# df_transformed = df_transformed.withColumn('first_above', first_above(df_transformed.features))
# df_transformed = df_transformed.withColumn('last_below', last_below(df_transformed.features))
# df_transformed = df_transformed.withColumn('time_max', time_max(df_transformed.features))

# %% language="spark"
#
# # Setup the pipeline
# # Build the vectors, keep the 5 daily counts pca features and add the 3 new features from above
# vec_assembler_time = VectorAssembler(inputCols=['first_above', 'last_below', 'time_max'], outputCol='time_features')
#
# # Normalize each vector to have unit 2-norm
# normalizer = Normalizer(inputCol="time_features", outputCol="normalized_features_v2")
#
# vec_assembler = VectorAssembler(inputCols=['pca_features_daily_count', 'normalized_features_v2'], outputCol="features_v2")
#
# pipeline = Pipeline(
#     stages=[
#         vec_assembler_time,
#         normalizer,
#         vec_assembler
#     ]
# )
#
# # Run the pipeline
# df_transformed = (pipeline.fit(df_transformed).transform(df_transformed)).cache()

# %% magic_args="-o df_scores" language="spark"
# from pyspark.sql import Row
#
# # Compute the silouhette scores for different values of k
# ks = list(range(3, 20))
# scores = silouhette_scores(df_transformed, 'features_v2', ks=ks)
#
# # Get the best k
# best_k = max(scores, key=itemgetter(1))[0]
#
# print('Best k : {0}'.format(best_k))
#
# # Get back the results locally, have to create a df...
# df_scores = map(lambda x: Row(*x), scores) 
# df_scores = spark.createDataFrame(df_scores, ['k', 'silouhette_v2'])

# %%
# Plot the scores for each value of k
plt.plot(df_scores['k'], df_scores['silouhette_v2'], marker='o')

plt.title('Silouhette scores for different $k$')
plt.xlabel('k')
plt.ylabel('Silouhette score')
plt.show()

# %% language="spark"
#
# # Compute the Kmeans with the best k
# df_all_kmeans, model_all = kmeans(df_transformed, 'features_v2', k=best_k)
#
# # Show the cluster sizes
# model_all.summary.clusterSizes

# %% language="spark"
#
# # Print to which cluster each of our hashtags belong to
# df_all_kmeans.filter(df_all_kmeans.hashtag.isin(hashtags)).select('hashtag', 'prediction').show(n=10)

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 0, n=15)

# %% [markdown]
# #### Let's visualize the result of the clustering after adding the new features

# %% magic_args="-o df_cluster_plot" language="spark"
#
# input_cols = [str(x) for x in range(1, 367)]   
# # Save the greatest daily count for each hashtag for visualization purposes
# df_cluster_plot = df_all_kmeans.withColumn('greatest', F.greatest(*input_cols))

# %%
plot_clusters(df_cluster_plot, sample_size=3, min_threshold=10000, max_threshold=20000)

# %% language="spark"
# plot_clusters_scatter(df_all_kmeans, "features_v2")
# %matplot plt

# %% [markdown]
# ---
# ### Features based on the spectrum of the daily counts time series
#
# Here we would like to characterize the hashtags based on the spectrum of their daily counts. The idea is to differentiate hashtags that represent events spanning a relatively longer period (e.g. the covid19) from the one representing a very "sharp" event in time (e.g. the Oscars). For these 2 examples, we clearly a different behavior in the daily counts from the plot of part 3.b. The Oscar event has a very sharp spike around the event whereas the covid19 is more spread out over the year.
#
# The DCT of the signal returns a Vector of the same length of the original signal. We therefore need to reduce again the dimensionality of this spectrum. We can do this while limiting the loss of information by running again a PCA, similarly to what we have done for the daily counts.
#
# The drawback when doing this is that we completely lose the notion of time, i.e. we can no more situate an event during the year.

# %% language="spark"
# from pyspark.ml.feature import DCT
#
# # Setup the pipeline
#
# # Compute Discrete Cosine Transform (DCT) of the daily counts signal
# dct = DCT(inverse=False, inputCol="features", outputCol="dct")
#
# # Normalize each vector to have unit 2-norm
# normalizer = Normalizer(inputCol="dct", outputCol="normalized_features_dct")
#
# # Project on first 5 principal components the dct features
# pca_dct = PCA(k=5, inputCol="normalized_features_dct", outputCol="pca_dct")
#
# # Build the vectors, keep the 5 daily counts pca features, the 3 features from previous part and add the 5 dct pca features
# vec_assembler = VectorAssembler(inputCols=['features_v2', 'pca_dct'], outputCol="features_v3")
#
# pipeline = Pipeline(
#     stages=[
#         dct,
#         normalizer,
#         pca_dct,
#         vec_assembler
#     ]
# )
#
# df_transformed = (pipeline.fit(df_transformed).transform(df_transformed)).cache()

# %% magic_args="-o df_scores" language="spark"
# from pyspark.sql import Row
#
# # Compute the silouhette scores for different values of k
# ks = list(range(3, 20))
# scores = silouhette_scores(df_transformed, 'features_v3', ks=ks)
#
# # Get the best k
# best_k = max(scores, key=itemgetter(1))[0]
#
# print('Best k : {0}'.format(best_k))
#
# # Get back the results locally, have to create a df...
# df_scores = map(lambda x: Row(*x), scores) 
# df_scores = spark.createDataFrame(df_scores, ['k', 'silouhette_v3'])

# %%
# Plot the scores for each value of k
plt.plot(df_scores['k'], df_scores['silouhette_v3'], marker='o')

plt.title('Silouhette scores for different $k$')
plt.xlabel('k')
plt.ylabel('Silouhette score')
plt.show()

# %% language="spark"
#
# # Compute the Kmeans with the best k
# df_all_kmeans, model_all = kmeans(df_transformed, 'features_v3', k=best_k)
#
# # Show the cluster sizes
# model_all.summary.clusterSizes

# %% language="spark"
#
# # Print to which cluster each of our hashtags belong to
# df_all_kmeans.filter(df_all_kmeans. hashtag.isin(hashtags)).select('hashtag', 'prediction').show(n=10)

# %% [markdown]
# We can see that we have a similar separation for our selected hashtags with the previous features.

# %% language="spark"
#
# # Show the content of the different clusters
# show_cluster(df_all_kmeans, 7, n=15)

# %% [markdown]
# #### Let's visualize the result of the clustering after adding the final features

# %% magic_args="-o df_cluster_plot" language="spark"
#
# input_cols = [str(x) for x in range(1, 367)]   
# # Save the greatest daily count for each hashtag for visualization purposes
# df_cluster_plot = df_all_kmeans.withColumn('greatest', F.greatest(*input_cols))

# %%
plot_clusters(df_cluster_plot, sample_size=3, min_threshold=10000, max_threshold=20000)

# %% language="spark"
# plot_clusters_scatter(df_all_kmeans, "features_v3")
# %matplot plt

# %% [markdown]
# # That's all, folks!

# %%

# %%

# %%
