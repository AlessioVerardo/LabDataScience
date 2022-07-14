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
#     display_name: PySpark
#     language: python
#     name: pysparkkernel
# ---

# %% [markdown]
# # DSLab Homework4 - More trains (PART III)
#
# ## Hand-in Instructions:
# - __Due: 11.05.2021 23:59:59 CET__
# - your project must be private
# - git push your final verion to the master branch of your group's Renku repository before the due date
# - check if Dockerfile, environment.yml and requirements.txt are properly written
# - add necessary comments and discussion to make your codes readable
#
# ## NS Streams
# For this homework, you will be working with the real-time streams of the NS, the train company of the Netherlands. You can see an example webpage that uses the same streams to display the train information on a map: https://spoorkaart.mwnn.nl/ . 
#
# To help you and avoid having too many connections to the NS streaming servers, we have setup a service that collects the streams and pushes them to our Kafka instance. The related topics are: 
#
# `ndovloketnl-arrivals`: For each arrival of a train in a station, describe the previous and next station, time of arrival (planned and actual), track number,...
#
# `ndovloketnl-departures`: For each departure of a train from a station, describe the previous and next station, time of departure (planned and actual), track number,...
#
# `ndovloketnl-gps`: For each train, describe the current location, speed, bearing.
#
# The events are serialized in JSON (actually converted from XML), with properties in their original language. Google translate could help you understand all of them, but we will provide you with some useful mappings.

# %% [markdown]
# ---
# **PART III is in PySpark kernel**
#
# With the pyspark kernel, code is executed on the spark driver by default. To execute locally, one has to use the %%local magic. The SparkContext is available as  `sc`. [More informations on the PySpark kernel](https://github.com/jupyter-incubator/sparkmagic/blob/master/examples/Pyspark%20Kernel.ipynb).

# %%
# %%local
ipython = get_ipython()
print('Current kernel: {}'.format(ipython.kernel.kernel_info['implementation']))

# %% [markdown]
# ---
# ## Set up environment
#
# Run the following cells below before running the other cells of this notebook. Run them whenever you need to recreate a Spark context. Pay particular attention to your `username` settings, and make sure that it is properly set to your user name, both locally and on the remote Spark Driver.
#
# Configure your spark settings:
# 1. name your spark application as `"<your_gaspar_id>-homework4"`.
# 2. make the required kafka jars available on the remote Spark driver.

# %%
# %%local
import os
import json

username = os.environ['RENKU_USERNAME']

configuration = dict(
    name = "{}-homework4".format(username),
    executorMemory = "1G",
    executorCores = 2,
    numExecutors = 2,
    conf = {
        "spark.jars.packages":"org.apache.spark:spark-streaming-kafka-0-8_2.11:2.3.2,org.apache.kafka:kafka_2.11:1.0.1"
    }
)

ipython = get_ipython()
ipython.run_cell_magic('configure', line="-f", cell=json.dumps(configuration))

# %% [markdown]
# Create a new session unless one was already created above (check for `✔` in current session)

# %%
# Initialize spark application

# %% [markdown]
# Send `username` to the Spark driver.

# %%
# %%send_to_spark -i username -t str -n username

# %%
print("You are {} on the Spark driver.".format(username))

# %% [markdown]
# ---

# %% [markdown]
# ## Create a Kafka client

# %%

from pykafka import KafkaClient
from pykafka.common import OffsetType

ZOOKEEPER_QUORUM = 'iccluster029.iccluster.epfl.ch:2181,' \
                   'iccluster044.iccluster.epfl.ch:2181,' \
                   'iccluster052.iccluster.epfl.ch:2181'

client = KafkaClient(zookeeper_hosts=ZOOKEEPER_QUORUM)

# %% [markdown]
# Working on data streams is often times more complex compared to using static datasets, so we will first look at how to create static RDDs for easy prototyping.
#
# You can find below a function that creates a static RDD from a Kafka topic.

# %%

from itertools import islice

def simple_create_rdd(topic, from_offset, to_offset):
    """Create an RDD from topic with offset in [from_offset, to_offest)."""
    
    consumer = client.topics[topic].get_simple_consumer(
        auto_offset_reset=OffsetType.EARLIEST if from_offset == 0 else from_offset - 1,
        reset_offset_on_start=True
    )
    
    return sc.parallelize((msg.offset, msg.value) for msg in islice(consumer, to_offset - from_offset))


# %% [markdown]
# To check this function, we need to retrieve valid offsets from Kafka.

# %%
topic = client.topics[b'ndovloketnl-arrivals']
topic.earliest_available_offsets()

# %% [markdown]
# Now, we can for example retrieve the first 1000 messages from the topic `ndovloketnl-arrivals`.

# %%
offset = topic.earliest_available_offsets()[0].offset[0]
rdd = simple_create_rdd(b'ndovloketnl-arrivals', offset, offset+1000)

# %%
rdd.first()

# %%
rdd.count()

# %% [markdown]
# ## Streams from Kafka

# %%
# Define the checkpoint folder
checkpoint = 'hdfs:///user/{}/checkpoint/'.format(username)
print('checkpoint created at hdfs:///user/{}/checkpoint/'.format(username))

# %%
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

# Create a StreamingContext with two working threads and batch interval of 5 seconds.
# Each time you stop a StreamingContext, you will need to recreate it.
ssc = StreamingContext(sc, 10)
ssc.checkpoint(checkpoint)

group_id = 'ns-{0}'.format(username)

# Input streams
arrival_stream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, group_id, {'ndovloketnl-arrivals': 1})
departure_stream = KafkaUtils.createStream(ssc, ZOOKEEPER_QUORUM, group_id, {'ndovloketnl-departures': 1})

# %% [markdown]
# For now, let's just print the content of the streams. Because we set the batch interval as 10 seconds and the timeout also as 10 seconds, you are supposed to see exactly one batch from each stream, like:
# ```
# -------------------------------------------
# Time: 2021-04-27 10:11:50
# -------------------------------------------
# <ONE_BATCH_OF_ARRIVAL_STREAM>
# ...
# -------------------------------------------
# Time: 2021-04-27 10:11:50
# -------------------------------------------
# <ONE_BATCH_OF_DEPARTURE_STREAM>
# ...
# ```
# **Note:** the output may be shown after you run `ssc.stop`.

# %%
arrival_stream.pprint(num=2) # print the first 2 messages
departure_stream.pprint(num=2) # print the first 2 messages

ssc.start()
ssc.awaitTermination(timeout=10)

# %%
ssc.stop(stopSparkContext=False, stopGraceFully=False)

# %% [markdown]
# You will need to adjust the batch interval (10 seconds here) in accordance with the processing times. Use the spark UI to check if batches are not accumulating.

# %% [markdown]
# ---

# %% [markdown]
# ## Part III - Live stopping time (10 points)

# %% [markdown]
# In this part, we will have a look at the two other streams, namely `ndovloketnl-arrivals` and `ndovloketnl-departures`. Each time a train arrives at or leaves a station, a message is generated. Let's have a look at the content.

# %%
import json
from pykafka.common import OffsetType

example_arrivals = client.topics[b'ndovloketnl-arrivals'].get_simple_consumer(
    auto_offset_reset=OffsetType.EARLIEST,
    reset_offset_on_start=True
).consume()
print(json.dumps(json.loads(example_arrivals.value), indent=2))

# %%
example_departures = client.topics[b'ndovloketnl-departures'].get_simple_consumer(
    auto_offset_reset=OffsetType.EARLIEST,
    reset_offset_on_start=True
).consume()
print(json.dumps(json.loads(example_departures.value), indent=2))

# %% [markdown]
# We can see that the messages have the following structure:
#
# ```
# {
#   'ns1:PutReisInformatieBoodschapIn': {
#     'ns2:ReisInformatieProductDVS' or 'ns2:ReisInformatieProductDAS': {
#       'ns2:DynamischeVertrekStaat' or 'ns2:DynamischeAankomstStaat': {
#           'ns2:RitStation': <station_info>,
#           'ns2:Trein' or 'ns2:TreinAankomst': {
#               'ns2:VertrekTijd' or 'ns2:AankomstTijd': [<planned_and_actual_times>],
#               'ns2:TreinNummer': <train_number>,
#               'ns2:TreinSoort': <kind_of_train>,
#               ...
#           }
#            
#       }
#     }
#   }
# }
# ```
#
# We can see also that the train stations have a long name `ns2:LangeNaam`, a medium name `ns2:MiddelNaam`, a short name `ns2:KorteNaam`, a station code `ns2:StationCode` and a kind of nummerical ID `ns2:UICCode`. When giving information about times, tracks, direction,... you will find sometimes the information twice with the status `Gepland` (which means planned, according to the schedule) and `Actueel`(which means the actual measured value). 

# %% [markdown]
#
#
# We want to compute the time a train stays at a station and get a real-time histogram for a given time window. To begin with, you need to write some parsing functions that will allow you to get information from the data streams. We have prepare one function `parse_train_dep` for the stream `ndovloketnl-departures`, which returns a Key-Value pair.

# %%
import json

def parse_train_dep(s):
    obj = json.loads(s)
    tn = (obj.get('ns1:PutReisInformatieBoodschapIn', {})
             .get('ns2:ReisInformatieProductDVS', {})
             .get('ns2:DynamischeVertrekStaat', {})
             .get('ns2:Trein', {})
             .get("ns2:TreinNummer"))
    st = (obj.get('ns1:PutReisInformatieBoodschapIn', {})
             .get('ns2:ReisInformatieProductDVS', {})
             .get('ns2:DynamischeVertrekStaat', {})
             .get('ns2:RitStation', {})
             .get("ns2:UICCode"))
    if tn and st:
        return [("{}-{}".format(tn, st), obj)]
    else:
        return []


# %%
parse_train_dep(example_departures.value)


# %% [markdown]
# ### a) Parse arrivals - 2/10
# Please check the function `parse_train_dep` above. Explain how we construct the Key and the Value, and why we construct them in this way.

# %% [markdown]
# The key is a concatenation of the train number and the unique code of the railway station in which it arrives or exits. 
# It is constructed in such a way so that we are able to quickly match the arrival and departure of a train from a station. 

# %% [markdown]
# ### b) Parse departures - 4/10
# Take `parse_train_dep` as an example and write the function `parse_train_arr` for the stream `ndovloketnl-arrivals`. Make sure they have the same output format.

# %%
def parse_train_arr(s):
    obj = json.loads(s)
    tn = (obj.get('ns1:PutReisInformatieBoodschapIn', {})
             .get('ns2:ReisInformatieProductDAS', {})
             .get('ns2:DynamischeAankomstStaat', {})
             .get('ns2:TreinAankomst', {})
             .get("ns2:TreinNummer"))
    st = (obj.get('ns1:PutReisInformatieBoodschapIn', {})
             .get('ns2:ReisInformatieProductDAS', {})
             .get('ns2:DynamischeAankomstStaat', {})
             .get('ns2:RitStation', {})
             .get("ns2:UICCode"))
    if tn and st:
        return [("{}-{}".format(tn, st), obj)]
    else:
        return []


# %%
parse_train_arr(example_arrivals.value)

# %% [markdown]
# ### c) Extract actual time - 4/10
# Another parsing function is `get_actual_time`, which will allow you to extract the **actual** time from the fields of time information, which are `ns2:AankomstTijd` in the arrival stream and `ns2:VertrekTijd` in the departure stream. 
#
# __Note:__ These two fields may be empty and they may not contain the actual time information. In both cases the function should return `None`.

# %%
import datetime
def get_actual_time(tab):
    date = None
    for field in tab: 
        if field.get('@InfoStatus') == u'Actueel':
            date = datetime.datetime.strptime(field.get(u'#text'), "%Y-%m-%dT%H:%M:%S.%fZ")
    return date


# %%
# Get the field of time in the departure stream
example_dep_json = json.loads(example_departures.value)
example_dep_tab = (example_dep_json.get('ns1:PutReisInformatieBoodschapIn', {})
                                   .get("ns2:ReisInformatieProductDVS", {})
                                   .get("ns2:DynamischeVertrekStaat", {})
                                   .get("ns2:Trein", {})
                                   .get("ns2:VertrekTijd",{}))

# %%
example_dep_tab

# %%
assert get_actual_time(example_dep_tab) == datetime.datetime(2022, 4, 26, 8, 24, 35)

# %%
# Get the field of time in the arrival stream
example_arr_json = json.loads(example_arrivals.value)
example_arr_tab = (example_arr_json.get('ns1:PutReisInformatieBoodschapIn', {})
                                   .get("ns2:ReisInformatieProductDAS", {})
                                   .get("ns2:DynamischeAankomstStaat", {})
                                   .get("ns2:TreinAankomst", {})
                                   .get("ns2:AankomstTijd", {}))

# %%
assert get_actual_time(example_arr_tab) == datetime.datetime(2022, 4, 25, 16, 25, 19)

# %%
