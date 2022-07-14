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
# # Stream processing with Spark and Kafka
#
# ----
# <div class="alert alert-danger">
#     <b>The current server configuration does not permit execution of cells after 2.1c. </b>
# </div>
#
# <div class="alert alert-block alert-warning"><u><b>A word of caution:</b></u>
# In the following exercises we are going to learn about data stream processing. It is a programming patterns that consists of at least one data producer and one data consumer working in tandem. In other words, that's at least two processes (and normally many more) running simultaneously and streaming data from one to the other in real-time. In the following exercises, we will be running producers and consumers in the same notebook. However, Jupyter notebooks can only run one cell at the time. Therefore, with the exception of the wiki-edits exercise, which connects to a data producer outside the notebook, we will alternate between producer and consumer cells. It works because the producer's messages are saved in Kafka, and the processing in the consumer is deferred until we stop the producer and start the consumer.  This is sufficient for the objectives of this tutorial, which is to familiarize yourselves with the Kafka API. Yet this back-and-forth should only be used in <i>debug mode</i>. The other option would to run the tutorial from separate notebooks, or not use notebooks, but we would miss the step-by-step learning approach. In a real world application, producers and consumers are running concurrently and much more efficiently, with consumers processing messages coming from one end of Kafka on the fly, while producers are submitting new messages to the other end.
# </div>
#
# ----

# %% [markdown]
# Run the 5 cells below before running the other cells of this notebook. Run them whenever you need to recreate a Spark context. **Pay particular attention** to your `username` settings, and make sure that it is properly set to your user name, both locally and on the remote Spark Driver.

# %% [markdown]
# ----
#
# Configure your spark environment, and make the required kafka jars available on the remote Spark driver.

# %%
# %load_ext sparkmagic.magics

# %%
import os
import json
from IPython import get_ipython

username = os.environ['RENKU_USERNAME']
server = "http://iccluster029.iccluster.epfl.ch:8998"

configuration = dict(
    name = "{}-exercises-w9".format(username),
    executorMemory = "1G",
    executorCores = 2,
    numExecutors = 2,
    conf = {
        "spark.jars.packages":"org.apache.spark:spark-streaming-kafka-0-8_2.11:2.3.2,org.apache.kafka:kafka_2.11:1.0.1"
    }
)

ipython = get_ipython()
ipython.run_cell_magic("spark", line="config", cell=json.dumps(configuration))

# %% [markdown]
# ----
# Create a new session unless one was already created above (check for `✔` in current session)

# %%
# Initialize spark application
get_ipython().run_line_magic(
    "spark", "add -s {0}-week9 -l python -u {1} -k".format(username, server)
)

# %%
# %%spark?

# %% [markdown]
# ## Part 2. Spark streaming

# %% [markdown]
# In the last module, you have experimented with Spark, processing batches of data in RDDs and dataframes. Spark also supports "streaming", namely it defines a stream as a sequence of mini-batches. You can find a pretty detailed introduction here: https://spark.apache.org/docs/2.3.2/streaming-programming-guide.html. Pay particular attention to the section [_points to remember_](https://spark.apache.org/docs/2.3.2/streaming-programming-guide.html#points-to-remember).

# %% language="spark"
# # from pyspark import SparkContext
# import os
# from pyspark.streaming import StreamingContext
# username = os.environ['TIMELINE_FLOW_NAME_TAG'].split('-')[0]
# # Create a StreamingContext with two working thread and batch interval of 2 second.
# # Each time you stop a StreamingContext, you will need to recreate it here.
# ssc = StreamingContext(sc, 2)
# ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))
# print('checkpoint created at hdfs:///user/{}/checkpoint/'.format(username))

# %% [markdown]
# You can open the Spark UI, following the link in the output above and you should notice a new tab (only when a stream is running), showing the processing of each mini-batch.

# %% [markdown]
# ### 2.1 From a RDD queue

# %% [markdown]
# We will start with a very simple stream, out of a list of RDDs generated manually. Here is an example of RDDs with a quite simple counter. 
#
# ```
# rdds = []
# for i in range(20):
#     rdds.append(sc.range(i * 100, (i + 1) * 100))
#
# ```
#
# **2.1.a** In the same vein, generate a list of RDDs with random numbers with a uniform distribution. (hint: have a look at the [spark MLlib package](https://spark.apache.org/docs/2.3.1/api/python/pyspark.mllib.html))

# %% language="spark"
# from pyspark.mllib.random import RandomRDDs
# rdds = []
#
# for i in range(20):
#     rdds.append(RandomRDDs.uniformRDD(sc, 1))

# %% [markdown]
# Now let's create the stream and print it from the stream processor. The python API for spark streaming has only a few options to get your results back. `pprint`is the most convenient for debugging and testing, `saveAsTextFiles` can be used for persisting the output and the most generic one is `foreachRDD` which can execute any function. ( https://spark.apache.org/docs/2.3.1/streaming-programming-guide.html#output-operations-on-dstreams )
#
# Once you start the stream computation with `ssc.start()`, you can check on the UI what is happening and you should see the output here after each batch. You will need to explicitely call stop `ssc.stop(stopSparkContext=False)`, which requires you to re-instanciate the streaming context if you want to start a new stream, or re-start this one.
#
# `ssc.awaitTermination(timeout=10)` terminates the stream and prints all the streaming output within the 10-second period. In real life, when stream is started by `ssc.start()`, we are able to check the outputs directly from standard output. However, in the context of sparkmagic notebook, we need to rely on `ssc.awaitTermination(...)` in order to "pack up" the outputs from the server side and print them all together.
#
# **2.1.b** Run the stream, it will stop after 10 seconds.

# %% language="spark"
# dstream = ssc.queueStream(rdds)
#
# # Print the elements of each RDD generated in this DStream to the console
# dstream.pprint()
# ssc.start()
# ssc.awaitTermination(timeout=10)

# %% [markdown]
# -----
# **2.1.c** Stop the stream context, but do not stop the Spark context yet.

# %% language="spark"
# ssc.stop(stopSparkContext=False)

# %% [markdown]
# <div class="alert alert-danger">
#     <b> The current server configuration does not permit execution of cells after 2.1c. </b>
# </div>

# %% [markdown]
# **Note** that the streams output a value at 2 seconds interval. This interval corresponds to the length of a micro-batch. It is configurable when you create the StreamContext.
#
# Now let's get back to our exercise and use the previoulsy generated random values to perform a random walk similar to [part I](./DSLab-w9-1.ipynb). The exercise demonstrates the capacity of Spark streaming to maintain a current state and apply updates to it. The state is backup-ed in your user folder on hdfs, in the checkpoint location that you specified when you created the streaming context. You can pass this checkpoint location when recreating the context to re-start from the point when you stopped it.
#
# Do not forget to stop your StreamContext (with `stopSparkContext=False`) when you are done. And always remember to recreate the StreamContext after you stop it before reusing it.
#
# **2.1.d** Restart the streaming context and apply some transformations on the stream of random data such that it outputs 2D coordinates of a random walk.

# %% language="spark"
# # Recreate the streaming context, because it has been stopped
# ssc = StreamingContext(sc, 2)
# ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))

# %% [markdown]
# Create a random walk function that takes as input a randome value (_randValue_) and a (X,Y) a current position tuple (_xyPosition_), converts the random value in one of the four directions, and return the new position tuple after moving from the current position toward the given direction. 

# %% language="spark"
# import math
#
# directions = [(0,1), (0, -1), (1, 0), (-1, 0)]
#
# def walk(randValue, xyPosition):
#     if xyPosition is None:
#         xyPosition = (0,0)
#     if randValue:
#         d = directions[int(math.floor(randValue[0] * 4))]
#         return (xyPosition[0] + d[0], xyPosition[1] + d[1])
#     else:
#         return None

# %% language="spark"
# dstream = ssc.queueStream(rdds)
# dstream = dstream.map(lambda x: (1, x)).updateStateByKey(walk)

# %% [markdown]
# **2.1.e** Modify the following output function such that it produces data that can then by plotted by the same function you wrote before in [notebook part I](DSLab-week9-part1.ipynb).

# %%
from pykafka import KafkaClient
from pykafka.common import OffsetType

ZOOKEEPER_QUORUM = 'iccluster029.iccluster.epfl.ch:2181,' \
                   'iccluster044.iccluster.epfl.ch:2181,' \
                   'iccluster052.iccluster.epfl.ch:2181'

def sendToKafka(iter):
    client = KafkaClient(zookeeper_hosts=ZOOKEEPER_QUORUM)
    with client.topics[('test-random-' + username).encode()].get_sync_producer() as producer:
        for record in iter:
            producer.produce("{},{}".format(*record[1]).encode())


# %% language="spark"
# dstream.foreachRDD(lambda rdd: rdd.foreachPartition(sendToKafka))

# %% [markdown]
# **2.1.f** Now you can start it and while it is running launch the plot cell too. Note that it will stop producing events after all the elements of rdds have been processed. You can stop the StreamContext at that point.
#
# Note: Here we do not need to call `ssc.awaitTermination(...)` because we do not call `dstream.pprint()` therefore there are no outputs; we want the stream computation to continue until we stop it explicitely.

# %% language="spark"
# ssc.start()  # Start the computation

# %% [markdown]
# You can run the same code as in [notebook part I](DSLab-week9-part1.ipynb) before for plotting the random graph. We copy it below for illustration purpose to maintain the chronological order of the exercises in this notebook, but we recommend you to run it from the other notebook.

# %%
import plotly.express as px
import plotly.graph_objects as go

fig = px.line(x=[0], y=[0], width=600, height=500)
fig.update_xaxes(title_text='X', range=(-10,10))
fig.update_yaxes(title_text='Y', range=(-10,10))
gofig = go.FigureWidget(fig)
data = gofig.data[0]
gofig

# %% tags=[]
from pykafka import KafkaClient
from pykafka.common import OffsetType
import numpy as np
import time

ZOOKEEPER_QUORUM = 'iccluster029.iccluster.epfl.ch:2181,' \
                   'iccluster044.iccluster.epfl.ch:2181,' \
                   'iccluster052.iccluster.epfl.ch:2181'

client = KafkaClient(zookeeper_hosts=ZOOKEEPER_QUORUM)

consumer_group = 'test-group-' + username

consumer = (client
    .topics['test-random-' + username]
    .get_simple_consumer(
        consumer_group=consumer_group, 
        auto_offset_reset=OffsetType.EARLIEST, 
        auto_commit_enable=True,
        reset_offset_on_start=False,
        auto_commit_interval_ms=2000
    )
)

# %%
try:
    n = 0
    for message in consumer:
        if message is not None:            
            xx, yy = map(int, message.value.decode().split(","))
            with gofig.batch_update():
                data.x = np.append(data.x, xx)[-20:]
                data.y = np.append(data.y, yy)[-20:]
            n=n+1
            time.sleep(0.1)
except KeyboardInterrupt:
    print("Consumer stopped after {0} messages".format(n))
    pass

# %% [markdown]
# Observe the trace plot evolve in the plot above (maybe very slowly), after a while you can interrupt the kernel manually, and stop the stream context below.

# %% language="spark"
# ssc.stop(stopSparkContext=False) # Don't forget to stop the stream when you are done

# %% [markdown]
# ### 2.2 From Kafka

# %% [markdown]
# Note: rerun the cells at the beginning of this notebook if you need to restart from a fresh context. Make sure that your `username` is properly copied to the remote spark driver if you do that.

# %% [markdown]
# Here, we will consume message from the `wiki-edits` topic. It contains the stream of events fetched from the wikimedia sites.
# More info: https://www.mediawiki.org/wiki/API:Recent_changes_stream

# %% language="spark"
# from pyspark.streaming.kafka import KafkaUtils

# %% language="spark"
# from pyspark.streaming import StreamingContext
# ssc = StreamingContext(sc, 10)
# ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))

# %% language="spark"
# ZOOKEEPER_QUORUM = 'iccluster029.iccluster.epfl.ch:2181,' \
#                    'iccluster044.iccluster.epfl.ch:2181,' \
#                    'iccluster052.iccluster.epfl.ch:2181'
#
# dstream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, 'test-wiki-group-' + username, {'wiki-edits': 1})

# %% language="spark"
# dstream.pprint()

# %% [markdown]
# Start the stream context, and let it run for 10 seconds.

# %% language="spark"
# ssc.start()
# ssc.awaitTermination(timeout=10)

# %% [markdown]
# Stop the stream context, do not stop the Spark context yet.

# %% language="spark"
# ssc.stop(stopSparkContext=False)

# %% [markdown]
# The messages from the `wiki-edits` topic are JSON strings. We can write a function to parse the JSON messages.
#
# Note: sometimes the messages are invalid json, so we will use `flatMap` to discard those.

# %% language="spark"
# import json
#
# def parse_wiki(doc):
#     try:
#         return [json.loads(doc)]
#     except ValueError:
#         return []

# %% [markdown]
# We can apply `parse_wiki` on the stream:

# %% language="spark"
# ssc = StreamingContext(sc, 10)
# ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))
# dstream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, 'test-wiki-group-' + username, {'wiki-edits': 1})

# %% language="spark"
# wiki_dstream = dstream.flatMap(lambda x: parse_wiki(x[1]))

# %% language="spark"
# wiki_dstream.pprint()

# %% language="spark"
# ssc.start()
# ssc.awaitTermination(timeout=10)

# %% language="spark"
# ssc.stop(stopSparkContext=False)

# %% [markdown]
# **2.2.a** Create a stream from `wiki-edits` which contains only the `type` and `wiki` fields.

# %% language="spark"
# ssc = StreamingContext(sc, 10)
# ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))
#
# dstream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, 'test-wiki-group-' + username, {'wiki-edits': 1})
#
# # Create a stream with only 'type' and 'wiki' fields -->
#
# wiki_dstream = dstream.flatMap(lambda x: parse_wiki(x[1]) )
#
# wiki_dstream2 = wiki_dstream.map(lambda x: {'type': x['type'], 'wiki': x['wiki']})
#
# wiki_dstream2.pprint()
#
# ssc.start()
# ssc.awaitTermination(timeout=10)
# ssc.stop(stopSparkContext=False)

# %% [markdown]
# ## 2.3 Window operations
#
# We have seen how to collect data into a stream on spark RDDs and how to apply simple transformations on them. Spark allows to perform window operations on streams.
#
# Documentation: https://spark.apache.org/docs/2.3.1/streaming-programming-guide.html#window-operations

# %% [markdown]
# Let's have a look at what we can get from the `wiki-edits`. We will perform operations on windows of size 10 minutes, every 2 seconds.
#
# As a simple example, here is how to count the number of messages for such a window.

# %% language="spark"
# ssc = StreamingContext(sc, 2)
# ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))
#
# dstream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, 'test-wiki-group-' + username, {'wiki-edits': 1})
# wiki_dstream = dstream.flatMap(lambda x: parse_wiki(x[1]))

# %% language="spark"
# # 600s = 10minutes, every 2 seconds
# wiki_counts = wiki_dstream.countByWindow(600, 2)

# %% language="spark"
# wiki_counts.pprint()

# %% language="spark"
# ssc.start()
# ssc.awaitTermination(timeout=10)

# %% language="spark"
# ssc.stop(stopSparkContext=False)

# %% [markdown]
# Look at the documentation to see how to use the window operations.
#
# **2.3.a** Create a stream that counts the actions done by a bot and those done by humans within the time window. The `bot` field contains a boolean value.
#
# **2.3.b** Create a stream that gets the number of actions by `type`. Also get the number of actions by `wiki`.
#
# **2.3.c** Send the above streams to kafka topics (example in **2.1**) and use that to dynamically plot:
# - the volume of actions by bots vs actions by humans.
# - the volume of actions by `type`, by `wiki`,
#
# Use `wiki-plot-bot-<username>`,  `wiki-plot-type-<username>` and `wiki-plot-wiki-<username>` for the respective topic names.

# %% language="spark"
# # TODO: "2.3.a Create stream and count actions by bot vs actions by humans, hint count by value and window on windows of size 10min every 2 seconds
# ssc = StreamingContext(sc, 2)
# ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))
#
# dstream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, 'test-wiki-group-' + username, {'wiki-edits': 1})
#
# wiki_dstream = dstream.flatMap(lambda x: parse_wiki(x[1]))
#
# wiki_count_bot = wiki_dstream.map(lambda x: 'bot' if x['bot'] else 'human').countByValueAndWindow(600, 2)
#
# wiki_count_bot.pprint()
#
# ssc.start()
# ssc.awaitTermination(timeout=10)
# ssc.stop(stopSparkContext=False)

# %% language="spark"
# # TODO: 2.3.b Create two streams, one that gets the number of actions by type and stream that get the number of actions by wiki. -->
# ssc = StreamingContext(sc, 2)
# ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))
#
# dstream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, 'test-wiki-group', {'wiki-edits': 1})
#
# wiki_dstream = dstream.flatMap(lambda x: parse_wiki(x[1]))
#
# wiki_count_type = wiki_dstream.map(lambda x: x['type']).countByValueAndWindow(600, 2)
# wiki_count_wiki = wiki_dstream.map(lambda x: x['wiki']).countByValueAndWindow(600, 2)
#
# wiki_count_type.pprint()
# wiki_count_wiki.pprint()
#
# ssc.start()
# ssc.awaitTermination(timeout=20)
# ssc.stop(stopSparkContext=False)

# %%

# TODO: 2.3.c Create stream and write stream to topics

ssc = StreamingContext(sc, 2)
ssc.checkpoint('hdfs:///user/{}/checkpoint/'.format(username))

dstream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, 'test-wiki-group-' + username, {'wiki-edits': 1})
wiki_dstream = dstream.flatMap(lambda x: parse_wiki(x[1])).cache()

wiki_count_type = wiki_dstream.map(lambda x: x['type']).countByValueAndWindow(600, 30)
wiki_count_wiki = wiki_dstream.map(lambda x: x['wiki']).countByValueAndWindow(600, 30)
wiki_count_bot = wiki_dstream.map(lambda x: 'bot' if x['bot'] else 'human').countByValueAndWindow(600, 30)


from pykafka import KafkaClient
from functools import partial

def sendToKafka(iter, topic):
    client = KafkaClient(zookeeper_hosts=ZOOKEEPER_QUORUM)
    with client.topics[('wiki-plot-{}-{}'.format(topic, username)).encode()].get_sync_producer() as producer:
        for record in iter:
            producer.produce("{},{}".format(*record).encode())
            

wiki_count_type.foreachRDD(lambda rdd: rdd.foreachPartition(partial(sendToKafka, topic="type")))
wiki_count_wiki.foreachRDD(lambda rdd: rdd.foreachPartition(partial(sendToKafka, topic="wiki")))
wiki_count_bot.foreachRDD(lambda rdd: rdd.foreachPartition(partial(sendToKafka, topic="bot")))

ssc.start()

# %%
# TODO: use plotly express to visualize real time bar plots of bot counts versus human wiki edits counts

# %%
from pykafka import KafkaClient
from pykafka.common import OffsetType

ZOOKEEPER_QUORUM = 'iccluster029.iccluster.epfl.ch:2181,' \
                   'iccluster044.iccluster.epfl.ch:2181,' \
                   'iccluster052.iccluster.epfl.ch:2181'

client = KafkaClient(zookeeper_hosts=ZOOKEEPER_QUORUM)

# %%
import plotly.express as px
import plotly.graph_objects as go

fig = px.bar(x=['bot', 'human'], y=[0, 0], width=600, height=500)
fig.update_xaxes(title_text='')
fig.update_yaxes(title_text='Counts')
gofig = go.FigureWidget(fig)
data = gofig.data[0]
gofig

# %%
import numpy as np
import time

consumer = client.topics[('wiki-plot-bot-{}'.format(username)).encode()].get_simple_consumer()

try:
    n = 0
    for message in consumer:
        if message is not None:     
            t, c = message.value.decode().split(",")
            if t == 'bot':
                data.y = np.array([int(c), data.y[1]])
            else:
                data.y = np.array([data.y[0], int(c)])
            n = n+1
            time.sleep(0.1)
except KeyboardInterrupt:
    print("Consumer stopped after {0} messages".format(n))
    pass

# %% language="spark"
# ssc.stop(stopSparkContext=False)

# %%
