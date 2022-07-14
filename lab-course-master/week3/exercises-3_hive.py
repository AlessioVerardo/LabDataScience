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
# # Fist Steps with HIVE
#
# The first questions in this workbook are the same as those found in [exercise 2](exercises-2_epd.py). However, we encourage you to review them as they are an opportunity to become familiar with the SBB/CFF data that you will use later in your assessed projects.
#
# <div class="alert alert-block alert-warning">
#     <b>Important remark! </b>The SBB data contains billions of records, and some of the queries in this exercise are computationally intensive. To minimize the load on our big data backend, we strongly recommend that you work in groups rather than running your queries individually.</div>
#
# ----
#
# We import standard python packages. We will need them later.

# %%
import os
import pandas as pd
pd.set_option("display.max_columns", 50)
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
# %matplotlib inline

# %% [markdown]
# # Get connected to Hive with `PyHive`
#
# [PyHive](https://github.com/dropbox/PyHive) is a open-sourced Python package which creates a Python interface for Hive. In this exercise, we will mainly use PyHive to interact with the data using hiveql queries.
#
# All the HiveQL command described in the rest of this notebook (in the `query` strings) can also be executed directly using the `beeline` command line from a Terminal. This command line is the Hive client, which has been configured in your notebook environment to connect to the Hive server under your credentials.
#
# > _Important Note:_ for the sake of simplicity, the cluster is configured to use basic security. The settings in your notebook environment are such that they should prevent you from accidentally impersonating other users and causing any damage to our distributed data storage. However, they can easily be bypassed - **do not attempt it**. There is nothing to be proven, and you will have to face the consequences when things go avry.
#
# Execute the cell below exactly as it is (do not modify it!), it will connect you to the Hive server from this notebook.

# %%
from pyhive import hive

# Set python variables from environment variables
username  = os.environ['RENKU_USERNAME']
hive_host = os.environ['HIVE_SERVER2'].split(':')[0]
hive_port = os.environ['HIVE_SERVER2'].split(':')[1]

# create connection
conn = hive.connect(host=hive_host,
                    #host="iccluster044.iccluster.epfl.ch",
                    port=hive_port,
                    username=username) 
# create cursor
cur = conn.cursor()

# %% [markdown]
# # Data from SBB/CFF/FFS
#
# Data source: <https://opentransportdata.swiss/en/dataset/istdaten>
#
# In this part, you will leverage Hive to perform exploratory analysis of data published by the Open Data Platform Swiss Public Transport. (<https://opentransportdata.swiss>).
#
# Format: the dataset is presented a collection of textfiles with fields separated by ';' (semi-colon). The textfiles have been compressed using bzip.
#
# Location: you can find the data on HDFS at the path `/data/sbb/istdaten`.
#
# The full description from opentransportdata.swiss can be found in <https://opentransportdata.swiss/de/cookbook/ist-daten/> in four languages. Because of the translation typos there may be some misunderstandings. We suggest you rely on the German version and use an automated translator when necessary. We will clarify if there is still anything unclear in class and Slack. Here are the relevant column descriptions:
#
# - `BETRIEBSTAG`: date of the trip
# - `FAHRT_BEZEICHNER`: identifies the trip
# - `BETREIBER_ABK`, `BETREIBER_NAME`: operator (name will contain the full name, e.g. Schweizerische Bundesbahnen for SBB)
# - `PRODUKT_ID`: type of transport, e.g. train, bus
# - `LINIEN_ID`: for trains, this is the train number
# - `LINIEN_TEXT`,`VERKEHRSMITTEL_TEXT`: for trains, the service type (IC, IR, RE, etc.)
# - `ZUSATZFAHRT_TF`: boolean, true if this is an additional trip (not part of the regular schedule)
# - `FAELLT_AUS_TF`: boolean, true if this trip failed (cancelled or not completed)
# - `HALTESTELLEN_NAME`: name of the stop
# - `ANKUNFTSZEIT`: arrival time at the stop according to schedule
# - `AN_PROGNOSE`: actual arrival time
# - `AN_PROGNOSE_STATUS`: show how the actual arrival time is calcluated
# - `ABFAHRTSZEIT`: departure time at the stop according to schedule
# - `AB_PROGNOSE`: actual departure time
# - `AB_PROGNOSE_STATUS`: show how the actual departure time is calcluated
# - `DURCHFAHRT_TF`: boolean, true if the transport does not stop there
#
# Each line of the file represents a stop and contains arrival and departure times. When the stop is the start or end of a journey, the corresponding columns will be empty (`ANKUNFTSZEIT`/`ABFAHRTSZEIT`).
#
# In some cases, the actual times were not measured so the `AN_PROGNOSE_STATUS`/`AB_PROGNOSE_STATUS` will be empty or set to `PROGNOSE` and `AN_PROGNOSE`/`AB_PROGNOSE` will be empty.

# %% [markdown]
# # First step with Hive

# %% [markdown]
# You are now connected on the remote Hive server from your notebook.
#
# You are ready to go and can start sending HiveQL commands to the server.
#
# All the commands needed for this exercises are described in the [Hive reference manual](https://cwiki.apache.org/confluence/display/Hive/LanguageManual).
#
# In the rest of this notebook, we will ask questions, and present a partial solution. You will need to replace all instances of `**TODO**` with the appropriate text before running the command.
#
# Create your database on hive. We will name it with your EPFL gaspar name, which we have saved in the `username` variable.
#
# We use the `location` property to instruct Hive to create the our database in the HDFS folder we created at the end of [exercises 1](./exercises-1_hdfs.md), under /user/_**username**_/hive.

# %%
### !!! STOP AND READ !!! ###
### This cell and next will drop and recreate your personal database
### - This is ok if this is your first week with Hive
### - But think twice if you have coming back in a few weeks and have
###   tables that take a lot of time to create in this database.
query = """
    drop database if exists {0} cascade
""".format(username)
cur.execute(query)

# %%
query = """
    create database {0} location "/user/{0}/hive"
""".format(username)
cur.execute(query)

# %% [markdown]
# **Q1**: Create an **external** Hive table. Hive will create a reference to the files located under `/data/sbb/bz2/istdaten/` on the HDFS storage, and apply a table schema on it, but it will not manage the file itself if the table is declared external. If you drop the table, only the definition in Hive is deleted, the content of `/data/sbb/bz2/istdaten` is preserved.
#
# Feel free to browse the content of `/data/sbb/bz2/istdaten` with the hdfs command line in a Terminal, and notice that all the data is compressed with bzip2 (bz2). This is ok, Hive knows how to handle it.
#
# You can use the command below to create the table, but you will need to change one line in the code in order to get it to work.
#
# See the [Hive DLL](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL) reference manual for more information about the meaning of these commands.

# %% [markdown]
# <div class="alert alert-block alert-warning">
#     <b>WARNING! </b> As a rule of thumb - if you create Hive tables from existing HDFS files, <b>always</b> make it <b>external</b>. Otherwise Hive will delete the HDFS files when you drop your table, which 99.99999% of the time is not what you want to do.</div>

# %%
query = """
    drop table if exists {0}.sbb_bz2
""".format(username)
cur.execute(query)

query = """
    create external table {0}.sbb_bz2(
        BETRIEBSTAG string,
        FAHRT_BEZEICHNER string,
        BETREIBER_ID string,
        BETREIBER_ABK string,
        BETREIBER_NAME string,
        PRODUKT_ID string,
        LINIEN_ID string,
        LINIEN_TEXT string,
        UMLAUF_ID string,
        VERKEHRSMITTEL_TEXT string,
        ZUSATZFAHRT_TF string,
        FAELLT_AUS_TF string,
        BPUIC string,
        HALTESTELLEN_NAME string,
        ANKUNFTSZEIT string,
        AN_PROGNOSE string,
        AN_PROGNOSE_STATUS string,
        ABFAHRTSZEIT string,
        AB_PROGNOSE string,
        AB_PROGNOSE_STATUS string,
        DURCHFAHRT_TF string
    )
    ## TODO: specify a line format here ##
    stored as textfile
    location '/data/sbb/bz2/istdaten'
""".format(username)
cur.execute(query)

# %% [markdown]
# Now verify that your table was properly created.
#
# **Do not** remove the limit, and **do not** use a very large limit!!

# %%
query = """
    select * from {0}.sbb_bz2 limit 5
""".format(username)
cur.execute(query)
for result in cur.fetchall():
    print(result)

# %% [markdown]
# **Notes**: Alternatively, for retrieving `select` results, we could also use the Pandas `pd.read_sql` method, which requires two arguments: the query `query` and the connection `conn` (not the cursor `cur`) and returns a dataframe for review. You can check [here](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.read_sql.html?highlight=read_sql) for more information about this function.

# %%
pd.read_sql(query, conn)

# %% [markdown]
# **Q2**: Note the first row above. What did we do wrong? Can you add a table property to the table creation query in order to solve the problem? (hint: use tblproperties). Never forget, the table must be declared external!

# %%
query = """
    drop table if exists {0}.sbb_bz2
""".format(username)
cur.execute(query)

query = """
        ## TODO - same table as before, with a table properties. ##
""".format(username)
cur.execute(query)

query = """
    select * from {0}.sbb_bz2 limit 5
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# ----
# # Let's try different formats now
#
# There are storing the same data under three other formats, CSV, Bz2 CSV, ORC and PARQUET.
#
# You have already created a Hive table using the Bz2 compressed CSV data.
#
# **Q3**:
#
# * Can you create the tables sbb_csv, sbb_orc, and sbb_parquet for the other three formats respectively?
#
# * For each time time how long it takes to count the number of rows with the magic function `%%time` (do not use `%%timeit` !!).
#
# * Which storage format has the best read performances timewise?
# * Compare the size versus performance tradeoffs using their storage format respective footprints (sizes in bytes) calculated in the previous exercise?

# %%
query = """
    drop table if exists {0}.sbb_csv
""".format(username)
cur.execute(query)

query = """
        ## TODO - create a CSV external table ##
""".format(username)
cur.execute(query)

# %%
# %%time
query = """
    select count(*)from {0}.sbb_csv
""".format(username)
pd.read_sql(query, conn)

# %%
query = """
    drop table if exists {0}.sbb_orc
""".format(username)
cur.execute(query)

query = """
        ## TODO - create an ORC external table ##
""".format(username)
cur.execute(query)

query = """
    select * from {0}.sbb_orc limit 5
""".format(username)
pd.read_sql(query, conn)

# %%
# %%time
query = """
    select count(*)from {0}.sbb_orc
""".format(username)
pd.read_sql(query, conn)

# %%
query = """
    drop table if exists {0}.sbb_parquet
""".format(username)
cur.execute(query)

query = """
        ## TODO - create a Parquet external table ##
""".format(username)
cur.execute(query)

query = """
    select * from {0}.sbb_parquet limit 5
""".format(username)
pd.read_sql(query, conn)

# %%
# %%time
query = """
    select count(*)from {0}.sbb_parquet
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# ----
# # Let's focus on only one day now

# %% [markdown]
# **Q4**: Can you create the hive-managed table `sbb_05_11_2018` for 05.11.2018 using one of the table you created earlier?
#
# **Notes:**
#
# * You must complete the exercises of notebook [exercises-1_hdfs.md](./exercises-1_hdfs.md) first.
#
# * There are several way this can be done. Here we suggest that you derive this new table from the one you just created earlier. Here is the manual for [Create/Drop/Truncate Table](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable).
#
# * We recommend that you use the table with the most efficient storage format (see above) for this operation.
#
# * This is one of the rare cases where we will not ask you to declare the table external.
#

# %%
query = """
    drop table if exists {0}.sbb_05_11_2018
""".format(username)
cur.execute(query)

query = """
    create table {0}.sbb_05_11_2018
    stored as parquet
     # TODO: create sbb_05_11_2018 from one of your earlier tables #
""".format(username)
cur.execute(query)

query = """
    select * from {0}.sbb_05_11_2018 limit 5
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# **Q5**: Check that you have only one day using `distinct`.
#
# Note: `distinct` is a reserved keywords for hive, you can find the usage here: [ALL and DISTINCT Clauses](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Select#LanguageManualSelect-ALLandDISTINCTClauses)

# %%
# %%time
query = """
    ## TODO ##
     from {0}.sbb_05_11_2018
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# **Q6**: How many stops can you find for the date 05.11.2018? Display the information by transport type (bus, train, etc.) and sort it in decreasing order of the number of stops. Here, please save the query result into a dataframe and **plot** the result in a proper way using matplotlib.
#
# Note: you can keep the German labels like 'Zug' for train. You should only use standard SQL group, order and count commands. For further reading see the distinction between [Group By](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+GroupBy) and [Sort By](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+SortBy).

# %%
query = """
    #TODO#
    from {0}.sbb_05_11_2018
    #TODO#
""".format(username)
df_type = pd.read_sql(query, conn)
df_type

# %% [markdown]
# Now show a bar plot of the results _df_type_ in matplotlib.

# %%
#TODO#

# %% [markdown]
# **Q7**: Create a new table, using the data you imported in [exercises-1_hdfs.md](./exercises-1_hdfs.md). Name it `{username}.sbb_27_02_2022` (where `{username}` is your gaspar id).

# %%
#TODO#

# %% [markdown]
# # Delay Headache

# %% [markdown]
# **Q8**: `AN_PROGNOSE_STATUS` and `AB_PROGNOSE_STATUS` indicates how the actual arrival/departure time is acquired by the system. According to the [data description](https://opentransportdata.swiss/de/cookbook/ist-daten/), there are five different cases (`UNBEKANNT`: UNKNOWN, `Leer`: Empty, `PROGNOSE`: FORECAST, `GESCHAETZT`: ESTIMATED, `REAL`: REAL).
#
# Please show us how `AB_PROGNOSE_STATUS` is distributed for all train (lower case `PRODUKT_ID` is zug) on Jan 2018 and Jan 2019.
#
# Note: You can use `like` for pattern matching to filter out the data of these two months. Details can be found here [Relational Operators](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-RelationalOperators).
#
# We recommend that you use the table with the most efficient (faster) storage format for this operation.

# %%
query = """
    select AB_PROGNOSE_STATUS as dstatus, count(*) as count
    from {0}.#TODO#
    where lower(PRODUKT_ID)
       #TODO#
    group by AB_PROGNOSE_STATUS
""".format(username)
pd.read_sql(query, conn)

# %%
query = """
    select AB_PROGNOSE_STATUS as dstatus, count(*) as count
    from {0}.sbb
    where lower(PRODUKT_ID)
       #TODO#
    group by AB_PROGNOSE_STATUS
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# **Q9**: Can you calculate the average delay for each type of trains that depatured from Lausanne train station on 05.11.2018? Then, do the same for train arriving at Lausanne train station.
# Hints: use the `unix_timestamp` functions.

# %%
# departure
query = """
    with Times as(
        select upper(VERKEHRSMITTEL_TEXT) as stype, 
               #TODO#
        from {0}.sbb_05_11_2018
        where HALTESTELLEN_NAME like 'Lausanne'
    )
    select stype, avg(actual - expected) as delay
    from Times
    where actual > expected
    group by stype
    order by delay desc
""".format(username)
pd.read_sql(query, conn)

# %%
# arrival
query = """
    with Times as(
        select upper(VERKEHRSMITTEL_TEXT) as stype, 
               #TODO#
        from {0}.sbb_05_11_2018
        where HALTESTELLEN_NAME like 'Lausanne'
    )
    select stype, avg(actual - expected) as delay
    from Times
    where actual > expected
    group by stype
    order by delay desc
""".format(username)
pd.read_sql(query, conn)

# %%
