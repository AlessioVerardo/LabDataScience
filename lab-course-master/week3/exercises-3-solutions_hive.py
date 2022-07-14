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
# All the HiveQL command described in the rest of this notebook (in the `query` strings) can be executed directly using the `beeline` command line from a Terminal. This command line is the Hive client, which has been configured in your notebook environment to connect to the Hive server under your credentials.
#
# > _Important Note:_ for the sake of simplicity, the cluster is configured to use basic security. The settings in your notebook environment are such that they should prevent you from accidentally impersonating other users and causing any damage to our distributed data storage. However, they can easily be bypassed - **do not attempt it**. There is nothing to be proven, and you will have to face the consequences when things go avry.
#
# Execute the cell below exactly as it is (do not modify it!), it will connect you to the Hive server from this notebook.

# %%
from pyhive import hive

# Set python variables from environment variables
username = os.environ['RENKU_USERNAME']
hive_host = os.environ['HIVE_SERVER2'].split(':')[0]
hive_port = os.environ['HIVE_SERVER2'].split(':')[1]

# create connection
conn = hive.connect(host=hive_host,
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
# Location: you can find the data on HDFS at the path `/data/sbb/bz2/istdaten`.
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
# We use the `location` property to instruct Hive to create the our database in the HDFS folder we created at the end of [exercises 1](./exercises-1-solutions_hdfs.md), under /user/_**username**_/hive.

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

query = """
    use {0}
""".format(username)
cur.execute(query)

# %% [markdown]
# **Q1**: Create an **external** Hive table. Hive will create a reference to the files located under `/data/sbb/bz2/istdaten/` on the HDFS storage, and apply a table schema on it, but it will not manage the file itself if the table is declared external. If you drop the table, only the definition in Hive is deleted, the content of `/data/sbb/bz2/istdaten` is preserved.
#
# Feel free to browse the content of `/data/sbb/bz2/istdaten` with the hdfs command line in a Terminal, and notice that all the data is compressed with bzip2 (bz2). This is ok, Hive knows how to handle it.
#
# You will need to change one line in the code below in order to get it to work.
#
# See the [Hive DDL](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL) reference manual.

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
    row format delimited fields terminated by ';'
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
# **Q2**: Note the first row above. What did we do wrong? Can you add a table property to the table creation query in order to solve the problem? (hint: use tblproperties)
#
# **Solutions:**
# * You must drop the headers - use the little documented `tbleproperties("skip.header.line.count="1")`
# * See other common properties available in [tbleproperties](https://cwiki.apache.org/confluence/display/hive/LanguageManual+DDL#LanguageManualDDL-CreateTable). Also different `SerDe` (Serializer, Deserializer) such as OpenCSV, ORC SerDe[[1](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+ORC),[2](https://orc.apache.org/docs/hive-config.html)], Json SerDe, and Parquet Hive SerDe, will have their own properties.

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
    row format delimited fields terminated by ';'
    stored as textfile
    location '/data/sbb/bz2/istdaten'
    tblproperties ("skip.header.line.count"="1")
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
#
#
# **Solutions:**
#
# * First you must find the location of the external tables. The hint is in the earlier exercises which points to the location of `/data/sbb/bz2/istdaten`. A bit of `hdfs dfs -ls /data/sbb` exploration show that that the other tables are respectively under the HDFS directories `/data/sbb/csv/istdaten`, `/data/sbb/orc/istdaten` and `/data/sbb/parquet/istdaten` respectively.
# * The **CSV** format is easy. The `create table` similar to the **Bz2** format shown earlier, but for the location which is `/data/sbb/csv`.
# * For **ORC** and **PARQUET** you need to understand the storage format specified by the `stored as` directive[[3](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-StorageFormatsStorageFormatsRowFormat,StorageFormat,andSerDe)].
#
# **Results:**
# * For your convenience we have included under each query, the output of the same query executed in a Terminal using the `beeline` command line interface. The command provides additional details showing the time spent in each task during the execution of the query.
# * The running times shown here are purely indicative - they will vary depending on the load at the time the command is being executed, and the ability of the application master to negotiate worker containers that are closest to the data.
# * All tables return __2,104,333,948__ rows.
# * Note that the total CPU time reported by `%%time` is (much) lower than the Wall time. This is because most of the work happens outside the docker container of this jupyter notebook.
# * We observe that **ORC** and **PARQUET** are an order of magnitude faster than **CSV** or **Bz2**. If we run the command a number of time, we note that **ORC** tends to be slightly faster than **PARQUET** (8seconds, versus 22seconds average), and **CSV** is faster than **Bz2** (105 seconds versus 140 seconds average). Remember that the **Bz2** compression saves storage space and network IO as seen in the previous exercise, however decompressing the data must be paid with CPU, and there is thus a tradeoff. In this case the uncompressed **CSV** format wins over **Bz2**.
# * **Hint:** you can get more info about the table using the Hive query `describe formatted db_name.table_name}`[[4](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-Describe)]

# %%
query = """
    drop table if exists {0}.sbb_csv
""".format(username)
cur.execute(query)

query = """
    create external table {0}.sbb_csv(
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
    row format delimited fields terminated by ';'
    stored as textfile
    location '/data/sbb/csv/istdaten'
    tblproperties ("skip.header.line.count"="1")
""".format(username)
cur.execute(query)

# %%
# %%time
query = """
    select count(*) from {0}.sbb_csv
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# Output of the `beeline` CLI (**CSV** format, **105.62** seconds):
#
# ```
# ----------------------------------------------------------------------------------------------
#         VERTICES      MODE        STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED  
# ----------------------------------------------------------------------------------------------
# Map 1 .......... container     SUCCEEDED    533        533        0        0       0       0  
# Reducer 2 ...... container     SUCCEEDED      1          1        0        0       0       0  
# ----------------------------------------------------------------------------------------------
# VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 105.88 s   
# ----------------------------------------------------------------------------------------------
# INFO  : Status: DAG finished successfully in 105.62 seconds
# INFO  : 
# INFO  : Query Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : OPERATION                            DURATION
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : Compile Query                           0.25s
# INFO  : Prepare Plan                            6.00s
# INFO  : Get Query Coordinator (AM)              0.00s
# INFO  : Submit Plan                             0.79s
# INFO  : Start DAG                               1.03s
# INFO  : Run DAG                               105.62s
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : 
# INFO  : Task Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  :   VERTICES      DURATION(ms)   CPU_TIME(ms)    GC_TIME(ms)   INPUT_RECORDS   OUTPUT_RECORDS
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  :      Map 1         101608.00      2,846,410         13,347   2,103,849,218            1,562
# INFO  :  Reducer 2          19566.00          5,430             72             533                0
# INFO  : ----------------------------------------------------------------------------------------------
# ...
# Time taken: 113.691 seconds
# +-------------+
# |     _c0     |
# +-------------+
# | 2104333948  |
# +-------------+
# 1 row selected (114.024 seconds)
# ```
# ----

# %%
query = """
    drop table if exists {0}.sbb_orc
""".format(username)
cur.execute(query)

query = """
    create external table {0}.sbb_orc(
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
    stored as orc
    location '/data/sbb/orc/istdaten'
""".format(username)
cur.execute(query)

query = """
    select * from {0}.sbb_orc limit 5
""".format(username)
pd.read_sql(query, conn)

# %%
# %%time
query = """
    select count(*) from {0}.sbb_orc
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# Output of the `beeline` CLI (**ORC** format, **6.93** seconds):
#
# ```
# ----------------------------------------------------------------------------------------------
#         VERTICES      MODE        STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED  
# ----------------------------------------------------------------------------------------------
# Map 1 .......... container     SUCCEEDED    235        235        0        0       0       0  
# Reducer 2 ...... container     SUCCEEDED      1          1        0        0       0       0  
# ----------------------------------------------------------------------------------------------
# VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 7.14 s     
# ----------------------------------------------------------------------------------------------
# INFO  : Status: DAG finished successfully in 6.93 seconds
# INFO  : 
# INFO  : Query Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : OPERATION                            DURATION
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : Compile Query                           0.17s
# INFO  : Prepare Plan                            0.25s
# INFO  : Get Query Coordinator (AM)              0.00s
# INFO  : Submit Plan                             0.18s
# INFO  : Start DAG                               0.58s
# INFO  : Run DAG                                 6.93s
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : 
# INFO  : Task Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  :   VERTICES      DURATION(ms)   CPU_TIME(ms)    GC_TIME(ms)   INPUT_RECORDS   OUTPUT_RECORDS
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  :      Map 1           4532.00        870,490         10,713   2,104,333,948              391
# INFO  :  Reducer 2           1896.00          3,150             55             235                0
# INFO  : ----------------------------------------------------------------------------------------------
# ...
# Time taken: 8.141 seconds
# +-------------+
# |     _c0     |
# +-------------+
# | 2104333948  |
# +-------------+
# 1 row selected (8.343 seconds)
# ```
# ----

# %%
query = """
    drop table if exists {0}.sbb_parquet
""".format(username)
cur.execute(query)

query = """
    create external table {0}.sbb_parquet(
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
    stored as parquet
    location '/data/sbb/parquet/istdaten'
""".format(username)
cur.execute(query)

query = """
    select * from {0}.sbb_parquet limit 5
""".format(username)
pd.read_sql(query, conn)

# %%
# %%time
query = """
    select count(*) from {0}.sbb_parquet
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# Output of the `beeline` CLI (**PARQUET** format, **23.02** seconds):
#
# ```
# ----------------------------------------------------------------------------------------------
#         VERTICES      MODE        STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED  
# ----------------------------------------------------------------------------------------------
# Map 1 .......... container     SUCCEEDED    214        214        0        0       0       0  
# Reducer 2 ...... container     SUCCEEDED      1          1        0        0       0       0  
# ----------------------------------------------------------------------------------------------
# VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 23.21 s    
# ----------------------------------------------------------------------------------------------
# INFO  : Status: DAG finished successfully in 23.02 seconds
# INFO  : 
# INFO  : Query Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : OPERATION                            DURATION
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : Compile Query                           0.27s
# INFO  : Prepare Plan                            0.28s
# INFO  : Get Query Coordinator (AM)              0.00s
# INFO  : Submit Plan                             0.20s
# INFO  : Start DAG                               0.57s
# INFO  : Run DAG                                23.02s
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : 
# INFO  : Task Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  :   VERTICES      DURATION(ms)   CPU_TIME(ms)    GC_TIME(ms)   INPUT_RECORDS   OUTPUT_RECORDS
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  :      Map 1          21153.00      2,865,190         26,347   2,104,333,948              321
# INFO  :  Reducer 2          13429.00          2,980              0             214                0
# INFO  : ----------------------------------------------------------------------------------------------
# ...
# +-------------+
# |     _c0     |
# +-------------+
# | 2104333948  |
# +-------------+
# 1 row selected (24.555 seconds)
# ```

# %% [markdown]
# ----
# # Let's focus on only one day now

# %% [markdown]
# **Q4**: Can you create the hive-managed table `sbb_05_11_2018` for 05.11.2018 using one of the table you created earlier?
#
# **Notes:**
#
# * You must complete the exercises of notebook [exercises-1-solutions_hdfs.md](./exercises-1-solutions_hdfs.md) first.
#
# * There are several way this can be done. Here we suggest that you derive this new table from the one you just created earlier. Here is the manual for [Create/Drop/Truncate Table](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable).
#
# * We recommend that you use the table with the most efficient storage format (see above) for this operation.
#
# * This is one of the rare cases where we will not ask you to declare the table external.
#
# **Solutions:**
#
# * Use your table {username}.sbb_orc, it is the most efficient storage format.
# * Select all the rows where `BETRIEBSTAG` is `05.11.2018` in the sbb_orc table.
# * Insert the output of the selection inside a new new Hive-managed table sbb_05_11_2018, stored as `parquet` or `orc`.

# %%
query = """
    drop table if exists {0}.sbb_05_11_2018
""".format(username)
cur.execute(query)

query = """
    create table {0}.sbb_05_11_2018
    stored as parquet
    as 
        select *
        from {0}.sbb_orc
        where BETRIEBSTAG = '05.11.2018'
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
     select distinct BETRIEBSTAG from {0}.sbb_05_11_2018
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# **Q6**: How many stops can you find for the date 05.11.2018? Display the information by transport type (bus, train, etc.) and sort it in decreasing order of the number of stops. Here, please save the query result into a dataframe and **plot** the result in a proper way using matplotlib.
#
# Note: you can keep the German labels like 'Zug' for train. You should only use standard SQL group, order and count commands. For further reading see the distinction between [Group By](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+GroupBy) and [Sort By](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+SortBy).

# %%
query = """
    select lower(PRODUKT_ID) as ttype, count(*) as stops
        from {0}.sbb_05_11_2018
        group by lower(PRODUKT_ID)
        order by stops desc
""".format(username)
df_type = pd.read_sql(query, conn)
df_type

# %% [markdown]
# Now show a bar plot of the results _df_type_ in matplotlib.

# %%
df_type.replace('', 'unknown', inplace=True)
df_type.plot('ttype', 'stops', kind='bar', legend=False, logy=True)
plt.xlabel('Type')
plt.ylabel('#Stops')
plt.show()

# %% [markdown]
# **Q7**: Create a new table using the data you imported in [exercises-1-solutions_hdfs.md](./exercises-1-solutions_hdfs.md). Name it `{username}.sbb_2022_02_27` (where `{username}` is your gaspar id).

# %%
query = """
    drop table if exists {0}.sbb_2022_02_27
""".format(username)
cur.execute(query)

query = """
    create external table {0}.sbb_2022_02_27(
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
    row format delimited fields terminated by ';'
    stored as textfile
    location '/user/{0}/work1'
    tblproperties ("skip.header.line.count"="1")
""".format(username)
cur.execute(query)

# %%
query = """
    select * from {0}.sbb_2022_02_27 limit 5
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# # Delay Headache

# %% [markdown]
# **Q8**: `AN_PROGNOSE_STATUS` and `AB_PROGNOSE_STATUS` indicates how the actual arrival/departure time is acquired by the system. According to the [data description](https://opentransportdata.swiss/de/cookbook/ist-daten/), there are five different cases (`UNBEKANNT`: UNKNOWN, `Leer`: Empty, `PROGNOSE`: FORECAST, `GESCHAETZT`: ESTIMATED, `REAL`: REAL).
#
# Please show us how `AB_PROGNOSE_STATUS` is distributed for all train (`PRODUKT_ID` is zug) on Jan 2018 and Jan 2019.
#
# Note: You can use `like` for pattern matching to filter out the data of these two months. Details can be found here [Relational Operators](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-RelationalOperators).
#
# We recommend that you use the table with the most efficient storage format for this operation.
#
# **Solution:**
#
# * Use your table {0}.sbb_orc, it is the most efficient storage format.

# %%
query = """
    select AB_PROGNOSE_STATUS as dstatus, count(*) as count
    from {0}.sbb_orc
    where lower(PRODUKT_ID) = 'zug'
      and BETRIEBSTAG like '__.01.2018'
    group by AB_PROGNOSE_STATUS
""".format(username)
pd.read_sql(query, conn)

# %%
query = """
    select AB_PROGNOSE_STATUS as dstatus, count(*) as count
    from {0}.sbb_orc
    where lower(PRODUKT_ID) = 'zug'
      and BETRIEBSTAG like '__.01.2019'
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
               unix_timestamp(ABFAHRTSZEIT, 'dd.MM.yyy HH:mm') as expected, 
               unix_timestamp(AB_PROGNOSE, 'dd.MM.yyy HH:mm:ss') as actual
        from {0}.sbb_05_11_2018
        where HALTESTELLEN_NAME = 'Lausanne'
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
               unix_timestamp(ANKUNFTSZEIT, 'dd.MM.yyy HH:mm') as expected, 
               unix_timestamp(AN_PROGNOSE, 'dd.MM.yyy HH:mm:ss') as actual
        from {0}.sbb_05_11_2018
        where HALTESTELLEN_NAME = 'Lausanne'
    )
    select stype, avg(actual - expected) as delay
    from Times
    where actual > expected
    group by stype
    order by delay desc
""".format(username)
pd.read_sql(query, conn)

# %%
