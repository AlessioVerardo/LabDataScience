# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stream processing with Kafka

# %% [markdown]
# ----
#
# <div class="alert alert-block alert-warning"><u><b>A word of caution:</b></u>
# In the following exercises we are going to learn about data stream processing. It is a programming patterns that consists of at least one data producer and one data consumer working in tandem. In other words, that's at least two processes (and normally many more) running simultaneously and streaming data from one to the other in real-time. In the following exercises, we will be running producers and consumers in the same notebook. However, Jupyter notebooks can only run one cell at the time. Therefore, with the exception of the wiki-edits exercise, which connects to a data producer outside the notebook, we will alternate between producer and consumer cells. It works because the producer's messages are saved in Kafka, and the processing in the consumer is deferred until we stop the producer and start the consumer.  This is sufficient for the objectives of this tutorial, which is to familiarize yourselves with the Kafka API. Yet this back-and-forth should only be used in <i>debug mode</i>. The other option would to run the tutorial from separate notebooks, or not use notebooks, but we would miss the step-by-step learning approach. In a real world application, producers and consumers are running concurrently and much more efficiently, with consumers processing messages coming from one end of Kafka on the fly, while producers are submitting new messages to the other end.
# </div>
#
#
# ----

# %% [markdown]
# ## Part 1. Message queue

# %% [markdown]
# In this section we will introduce Apache Kafka ( https://kafka.apache.org/ ). Kafka is a distributed message broker were you can subscribe for a topic and receive a stream of messages. You will also see how to publish messages. Let's start with the initial setup.
#
# You will be using the pykafka package ( http://pykafka.readthedocs.io/en/latest/ ). The pykafka API is more pythonish and easier to use than other available packages [kafka-python](https://github.com/dpkp/kafka-python), and [confluent-kafka-python]().

# %% vscode={"languageId": "python"}
from pykafka import KafkaClient
import os

username = os.environ['RENKU_USERNAME']

ZOOKEEPER_QUORUM = 'iccluster029.iccluster.epfl.ch:2181,' \
                   'iccluster044.iccluster.epfl.ch:2181,' \
                   'iccluster052.iccluster.epfl.ch:2181'

client = KafkaClient(zookeeper_hosts=ZOOKEEPER_QUORUM)

# %% [markdown]
# **1.a** Look at the pykafka documentation and list the topics currently available:

# %% vscode={"languageId": "python"}
print(client.topics)

# %% [markdown]
# As you can see there are already some topics there and unfortunately no separation per user. So, be respectful with the work of your colleagues and with the infrastructure we put in place. More specifically, never write in a topic that you didn't create yourself or your teammate (if not expressly asked to). Be also carefull not to create infinite loops by publishing in the same topic you subscribed, without filtering or exit conditions.
#
# **1.b** That being said, let's create our own topic. Don't forget to make it unique with your username.

# %% vscode={"languageId": "python"}
name = ('test-topic-' + username)
tt = client.topics[name]

# %% [markdown]
# Kafka is configured to autocreate topics if they don't exist a priori. You can check in the list of topics, it should be there now. 
#
# **1.c** To write a message to the topic you need to instanciate a Producer. For simplicity and as we won't deal with very large (or very fast) data, you can use a synchronous publisher. Then, send a series of 10 different messages (for example "Hello world {i}!" with i from 1 to 10). You might need to encode the string to utf-8 before sending:

# %% vscode={"languageId": "python"}
with tt.get_sync_producer() as producer:
    for i in range(10):
        # TODO: Procuce hello world messages

# %% [markdown]
# **1.d** Now let's check what is in our topic and subscribe to it. Again, for simplicity reason, you can use the basic ```get_simple_consumer```. Notice that the reading loop never finishes, streams are infinite! You must loop for ever over the stream and interrupt it manually. You may want to catch the KeyboardInterrupt to avoid polluting the output with a stacktrace: Hint: use _consumer_ like a python generator.

# %% vscode={"languageId": "python"}
consumer = tt.get_simple_consumer()
# TODO: Consume the messages in a loop

# %% [markdown]
# If you run the previous cell several times you will notice that it always restarts from the beginning by default. We have configured Kafka to persist the data for a quite long time. So be also careful not to leave data producers running uselessly.
#
# This is useful for testing and re-running the same pipeline several time. But in production setting, you may want to remember the last message seen and continue from that. Luckily, Kafka can do it for you if you define a consumer group and commit the current offset (see the documentation: http://pykafka.readthedocs.io/en/latest/usage.html#consumer-patterns ). You can either set the auto_commit_enable, or manually call the commit_offsets method.
#
# **1.e** Rewrite the above consummer loop to only show the newest elements every-time it is run. You can modify and re-run the producing loop to add some more messages.
#
# * Use the earlieast offset
# * Auto commit
# * Reset offset at start
# * Auto commit at 2 seconds intervals

# %% vscode={"languageId": "python"}
# Alternate between this cell and producer cell [4], multiple times.
# Repeat with other values of OffsetType, i.e. OffsetType.LATEST
from pykafka.common import OffsetType

#TODO: Create a consumer with its offset set to the oldest undread message 
consumer = tt.get_simple_consumer(...)

# Consume and print the messages.
# You must run the "Hello world" to add some more messages each time you execute this code
try:
    for message in consumer:
        if message is not None:
            print(message.offset, message.value)
except KeyboardInterrupt:
    print("Consumer stopped.")
    pass

# %% [markdown]
# We will now generate a little bit more data and see how to make a live plot. We can use another topic, like `b"test-random-<insert you username>"`.
#
# **1.f** First write an infinite [python generator](https://wiki.python.org/moin/Generators) (or just a for loop) that outputs the 2D coordinates of a random walk
#
# * Here you can simply take a random walk on a grid, by taking a random jump of length 1 in any of the 4 directions at each step (hint: use [random.choice()](https://docs.python.org/3/library/random.html#random.choice)).
# * Produce a stream of Kafka string messages, of integer coordinate tuples of the form _X,Y_
# * Start at coordinates _X,Y=0,0_
# * Add a `time.sleep(0.1)` each time after you publish it to Kafka in order to control the throughput.
# * Run the producer for a few seconds, then stop it.

# %% vscode={"languageId": "python"}
from random import choice
import time

tr = client.topics[('test-random-' + username)]


def random_walk():
    """Random walk on a 2D grid, yielding coordinates in an infinite loop."""
    ...
    while True:
        ...
        yield position

    
with tr.get_sync_producer() as producer:
    try:
        n = 0
        for p in random_walk():
            # TODO: Produce the coordinates
            ...
    except KeyboardInterrupt:
        print("Producer stopped after {0} messages".format(n))
        pass

# %% [markdown]
# **1.g** Now we will use the [line plot](https://plotly.com/python/line-and-scatter/) and [chart events](https://plotly.com/python/chart-events/) from the plotly library. As seen in earlier exercises, it allows for live update of the graph on the notebook, pushing data from the kernel. First you need to create a line plot with a data source initialized with empty lists for x and y. Then, in the consuming loop, you will parse the message and update the data of the source. Make sure your code would work for a long running job without eating all the memory ;-)

# %% vscode={"languageId": "python"}
import plotly.express as px
import plotly.graph_objects as go

fig = px.line(x=[0], y=[0], width=600, height=500)
fig.update_xaxes(title_text='X', range=(-10,10))
fig.update_yaxes(title_text='Y', range=(-10,10))
gofig = go.FigureWidget(fig)
data = gofig.data[0]
gofig

# %% vscode={"languageId": "python"}
import numpy as np

consumer_group=('test-group-' + username)
consumer = client.topics[('test-random-' + username)] \
                 .get_simple_consumer(consumer_group=consumer_group, 
                                      auto_offset_reset=OffsetType.EARLIEST, 
                                      auto_commit_enable=True, reset_offset_on_start=False, auto_commit_interval_ms=2000)

try:
    n = 0
    for message in consumer:
        if message is not None:            
            xx, yy = map(int, message.value.decode().split(","))
            with gofig.batch_update():
                # TODO: update coordinates in data
                ...
            n=n+1
            time.sleep(0.5)
except KeyboardInterrupt:
    print("Consumer stopped after {0} messages".format(n))
    pass

# %% [markdown]
# Note that in a notebook you can only run one cell at the time. You thus must stop the producer before you can run the consumer. If you want to play with the producer and the consumer at the same time, feel free to copy one of them to another notebook or in a python script to be able to run both of them simultaneously. Do not forget to include the cells needed initialize the kafka client as well. If you are working with a teammate (e.g. one is producer and the other is the consumer), make sure that you agree on using the same topic name.

# %% [markdown]
# ----
# That's all folks!
#
# You can now move to [notebook part II](DSLab-w9-2.ipynb), and experience with Kafka in Spark.

# %% vscode={"languageId": "python"}
