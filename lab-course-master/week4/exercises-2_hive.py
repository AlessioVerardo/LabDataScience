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
# # Data Wrangling with HDFS, Hive and HBase
#
# The purpose of this exercise is to experiment with Hadoop-driven methods to perform the _data wrangling_ steps of the data science process: acquisition, munging, exploration, visualization, and statistical models training.
#
# On a standard Hadoop big-data stack cluster, most of those steps can be done using Hive. The same methods work with other equivalent technologies such as Presto, or AWS Athena. As we will see they are extremely useful to prepare your data for operations in a large data lake.
#
# In this notebook you will first put some data in Hive table to explore it. You will then convert it to a more efficient storage format for querying.
#
# Generally the process goes through the following sequence of Extract Transform Load steps:
#
# 1. Connect to Hive
# 2. Create a Hive database
# 3. Create (declare) the input Hive tables and [load](https://cwiki.apache.org/confluence/display/Hive/Tutorial#Tutorial-LoadingData) data into tables
#     * In this step you describe the schema, that is the name and type of each column of the table. You may also need to describe the _Serialization/Deserialization_ method (SerDe) used to express the raw files into this schema, such as comma separated values or JSON parsing directives.
#     * Note that you can either create an external table that points to a specified location within HDFS, or other storage locations such as HBase, in which data has been copied beforehand. Or you can create an empty table and then load the data from a local or an HDFS files directly into the Hive table.
# 4. Create (declare) an empty output Hive table
# 5. Query and transform Hive tables using Hive Query Language
# 6. Save your results into new Hive Table
#
#
# Part I. of this notebook takes you through all of the above steps in the most simple scenario where the methods of converting from raw file format to table schema can be automatically inferred from the structure of the input data (e.g. CSV files).
#
# Part II. describes the serialization and deserialization methods used to tell Hive how to transform raw data formats into the prescribed table schemas when Hive cannot automatically infer said transformation.
#
# Part III. shows the integration Hive on top of HBase, as an alternative to the standard Hive on top of HDFS storage seen in part I and II.
#
# While working on this notebook, you may find it easier to explore the data using bash commands. In order to help you with that we provide the helper notebook [exercises-1_cli.md](./exercises-1_cli.md).
#
# 📝 Note: as you go along, you will be using three categories of Hive commands:
# * Data Definition Language ([DDL](https://cwiki.apache.org/confluence/display/Hive/LanguageManual%20DDL)) commands - are used to create and modify Hive databases and tables, and other Hive objects.
# * Data Modification Language ([DML](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DML)) commands - are used to add, delete or update data in Hive table.
# * Data [retrieval](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Select) commands - are used to query the content of Hive tables.
#
#
# 📝 Note: as in most SQL implementations, HiveQL kewords are **case insensitive**. In the following we conform to the generally accepted SQL convention, and use upper case for SQL keywords, and lower case for user-defined table and column names.

# %% [markdown]
# -----
# Python Environment Initialization

# %%
import os
import pandas as pd
pd.set_option("display.max_columns", 50)
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

username = os.environ['RENKU_USERNAME']
hiveaddr = os.environ['HIVE_SERVER2']
print("Operating as: {0}".format(username))
print("Hive server at: {0}".format(hiveaddr))

# %% [markdown]
# ----
# We have been interacting with HDFS using its shell command line interface. You can also access HDFS progmmatically using the `hdfs3` python APIs. The package is already installed and ready to be used.
#
# See the [hdfs3 documentation](https://hdfs3.readthedocs.io/en/latest/api.html#hdfs3.core.HDFileSystem).

# %% [markdown]
# -----
# ## Part I - HDFS / Hive Storage format
#
# Hive gives you the ability to choose between many storage formats for your data. You can find a list in the [Hive Storage Formats](https://cwiki.apache.org/confluence/display/Hive/FileFormats) documentation ([references](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-StorageFormats))
#
# Among them, the most commonly used formats are _textfile_, _parquet_ and _Optmized Row Columnar (ORC)_.
#
# The _textfile_ format should be used mostly for external files that are already in text format, or to make the tables available as external files to machines or humans who are not able to process other formats. They are mostly seen at the _input_ and _output_ end points of your big-data processing pipeline, which is where the data comes in, or out, and where interoperability with external sources or data consumers is required.
#
# In other situations, we will prefer the _parquet_ or _ORC_ format, which are optimized to store Hive data more efficiently. In particular, _parquet_ or _ORC_ should be your first choice for _Hive-managed_ tables, or temporary tables.
#
# We have been able to compare the various storage format in exercises of [week3](../week3/exercises-3_hive.py). In most exercises the tables had been created for you with the desired storage format.
#
# In the next section we will learn how to create new tables in the desired format.
#
# #### Converting from _textfile_ to _ORC_
#
# When receiving a file in a text format, sometimes it makes sense to create a copy using the more efficient storage formats _parquet_ or _ORC_. Transforming the data into these formats has a (CPU) cost, however it only needs to be paid once, when the tables are created, and it is largely paid back by the read performance boost we get from them. Furtheremore, those formats are understood by many other utilities of the big data platform, such as Spark. We thus want to store this data in an _external_ tables, in a location that is accessible to those utilities.
#
# We illustrate the format conversion in 3 steps in the next exercise.

# %% [markdown]
# ------
# We must first create a connection to the Hive server.

# %%
from pyhive import hive

# Hive host and port number
(hive_host, hive_port) = hiveaddr.split(':')

# create connection
conn = hive.connect(host=hive_host, 
                    port=hive_port,
                    username=username) 
# create cursor
cur = conn.cursor()

# %% [markdown]
# -----
# Assume you have received the sbb data in CSV format and that this data is now stored in HDFS. In this exercise we use a subset of the sbb data (December 2020), which you can find under `/data/sbb/istdaten/2020/12` on HDFS. You can verify the content of the files using the command of the companion bash notebook [exercises-4_cli (A.)](exercises-4_cli.md).
#
# **Step 1.** Create an [_external table_](https://cwiki.apache.org/confluence/display/Hive/Managed+vs.+External+Tables) stored in _textfile_ format, located on `/data/sbb/istdate/2019/12`:
#
# * Create a database using your name
#
# * Make the database your default with `use`.
#
# * Drop the table if it exists
#
# * Create the table

# %%
query = """
    CREATE DATABASE IF NOT EXISTS {0}
""".format(username)
cur.execute(query)

query = """
    USE {0}
""".format(username)
cur.execute(query)

# %%
query = """
    DROP TABLE IF EXISTS {0}.sbb_csv_2020_12
""".format(username)
cur.execute(query)

# %%
query = """
    CREATE EXTERNAL TABLE {0}.sbb_csv_2020_12(
        betriebstag STRING,
        fahrt_bezeichner STRING,
        betreiber_id STRING,
        betreiber_abk STRING,
        betreiber_name STRING,
        produkt_id STRING,
        linien_id STRING,
        linien_text STRING,
        umlauf_id STRING,
        verkehrsmittel_text STRING,
        zusatzfahrt_tf STRING,
        faellt_aus_tf STRING,
        bpuic STRING,
        haltestellen_name STRING,
        ankunftszeit STRING,
        an_prognose STRING,
        an_prognose_status STRING,
        abfahrtszeit STRING,
        ab_prognose STRING,
        ab_prognose_status STRING,
        durchfahrt_tf STRING
    )
    ROW FORMAT DELIMITED FIELDS TERMINATED BY ';'
    STORED AS TEXTFILE
    LOCATION '/data/sbb/csv/istdaten/2020/12/'
""".format(username)
cur.execute(query)

# %% [markdown]
# Oops, we forgot to skip the headers again. No worries, there is no need to drop the table. We can modify it with the `alter table set tblproperties` [command](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-AlterTableProperties).

# %%
query ="""
ALTER TABLE {0}.sbb_csv_2020_12 SET TBLPROPERTIES ("skip.header.line.count"="1")
""".format(username)
cur.execute(query)

# %% [markdown]
# Verify external table `sbb_csv_2020_12`

# %%
query = """
SELECT * FROM {0}.sbb_csv_2020_12 LIMIT 5
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# ----
# **Step 2.** Create a new table in your HDFS folder, under `/user/{0}/hive/sbb/orc`.
#
# Notes:
# * We store this data in [ORC](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+ORC#LanguageManualORC-HiveQLSyntax) storage format
# * We use `TBLPROPERTIES` to set the compression mode to `SNAPPY`
# * The table is external. If we drop the table, the generated _ORC_ files will still be available to other big data applications.
# * It is a new table, and it is empty. We will insert data into it.
# * Hive will create the folder at the specified location on HDFS if it does not exist. Because we are all going to create this new table, we do not want to write over each other data, we will therefore locate this external table into our HDFS home folders.

# %%
query = """
    DROP TABLE IF EXISTS {0}.sbb_orc_2020_12
""".format(username)
cur.execute(query)

# %%
query = """
    CREATE EXTERNAL TABLE {0}.sbb_orc_2020_12(
        betriebstag STRING,
        fahrt_bezeichner STRING,
        betreiber_id STRING,
        betreiber_abk STRING,
        betreiber_name STRING,
        produkt_id STRING,
        linien_id STRING,
        linien_text STRING,
        umlauf_id STRING,
        verkehrsmittel_text STRING,
        zusatzfahrt_tf STRING,
        faellt_aus_tf STRING,
        bpuic STRING,
        haltestellen_name STRING,
        ankunftszeit STRING,
        an_prognose STRING,
        an_prognose_status STRING,
        abfahrtszeit STRING,
        ab_prognose STRING,
        ab_prognose_status STRING,
        durchfahrt_tf STRING
    )
    ROW FORMAT DELIMITED FIELDS TERMINATED BY ';'
    STORED AS ORC
    LOCATION '/user/{0}/hive/sbb/orc'
    TBLPROPERTIES ("orc.compress"="SNAPPY")
""".format(username)
cur.execute(query)

# %% [markdown]
# -----
# **Step 3.** This new table is currently empty. We can now upload the data from our other table into this new table, using the [insert overwrite](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DML#LanguageManualDML-InsertingdataintoHiveTablesfromqueries) command. This command may take a minutes to complete.

# %%
query = """
INSERT OVERWRITE TABLE {0}.sbb_orc_2020_12 SELECT * FROM {0}.sbb_csv_2020_12 
""".format(username)
cur.execute(query)

# %% [markdown]
# Verify that the content of the `sbb_orc_2020_12_01` table is similar to the content of the `sbb_csv_2020_01` table. Note that the ordering of the tables can be different. Since we did not impose a particular ordering (order by, sort by), the ordering varies depending on underlying storage formats, and can be arbitrary.

# %%
query = """
SELECT * FROM {0}.sbb_orc_2020_12 LIMIT 5
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# Voila. You have a created table stored in _ORC_ format. Because the table is external, dropping this table in Hive will not delete the ORC files, and you can reuse them in other Hive tables, or in your Spark applications, etc.
#
# **Note**: The Hive [load](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DML#LanguageManualDML-Loadingfilesintotables) command can also be used to load external files into Hive tables. Note however, that prior to Hive 3.0, this command did a pure file copy/move and it could not be used to transform _textfile_ to _orc_. Since Hive 3.0 onwards, `load` is internally rewritten into an `insert as select`, which allows additional load operations. However, you must include [SerDe](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-RowFormats&SerDe) (Serialize/Deserialize) instructions to tell Hive how to parse the plain text input files and match its content to the schema of the destination table. _SerDe_ are briefly discussed in the next exercises.
#
# **Note**: As discussed in class, the Hive convert the query into a MapReduce job. At any point in time, you may run the command [EXPLAIN](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Explain) to display the plan.
#
# E.g.:

# %%
cur.execute("EXPLAIN EXTENDED SELECT * FROM {0}.sbb_orc_2020_12 LIMIT 5".format(username))
cur.fetchall()

# %% [markdown]
# ----
# #### Concluding remarks
#
# Queries on the `sbb` tables stored as _ORC_ are an order of magnitude more efficient than queries on _textfile_ format, and they should therefore be used whenever possible.
# You can find the full 2018 to 2021 sbb data stored as _ORC_ under the `/data/sbb/orc/istdaten` HDFS folder.
#
# As of Hive 3.0.0, `ORC` is the only format that supports [ACID](https://cwiki.apache.org/confluence/display/Hive/Hive+Transactions) (Atomicity, Consistency, Isolation, Durability) and transactions in Hive.
#
# Feel free to repeat the above procedure, and experiment with other storage formats and compression algorithms, such as `stored as parquet tblproperties("parquet.compression"="ZLIB")` and compare their performances. Make sure that you keep your work under your `/user/{0}/` HDFS folder, and do not modify the content of the `/data/` HDFS folder. Also, to keep the computational load light on our big data cluster, do not use more than a month of data when experimenting with this feature.
#
# Finally, there are other optimization methods that we did not cover here. One can for instance use table [partitions](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-PartitionedTables), and clustering, to fine tune the structure of the Hive tables to better fit its underlying data and improve the performances of certain kind of queries.
#

# %% [markdown]
# ----
# ## Part II - Hive Serialization/Deserialization Format
#
# In the next set of exercises we review the methods to used to move external data in and out of Hive tables.
#
#
# <div class="alert alert-warning" style="margin: 10px">
# <strong>Please READ CAREFULLY</strong>
# <p><p>
# In the following exercises, and whenever you are using Twitter data, look twice and never forget to create your tables  <strong>EXTERNAL</strong>. If you do forget to declare your table external you will delete all our twitter files from HDFS as soon as you drop the tables and it will take us hours to put them back.
# </div>

# %% [markdown]
# **Step 1.** Import one day of Twitter data into Hive.
#
# Create an external table from HDFS dir `/data/twitter/json/` and call it **_username_.**`twitter`. The table should have a single column named json of type string. Do not forget to use your database for the table instead of the Hive default.
#
# A few hints:
# 1. You can explore the `/data/twitter/json/` folder using the command of the companion bash notebook [exercises-1_cli (B.)](exercises-1_cli.md). You will notice that the files are in bzip2 format. Do not worry about that, Hive knows how to handle compressed text files automatically.
# 2. The files have only one field per line
# 3. If you do not specify the row format, the default format fields terminated by '\n' will be used.
#
#
# **More about Hive partitioning:**
#
# Note the particular naming convention used in our HDFS folder structure, `year=.../month=.../day=...`. As mentioned earlier if we know the range of dates we want to query in advance, we can play with the `LOCATION` attribute of the table in order to create a smaller external table that contains only the files of a particular year, month and day of interest, such as `LOCATION='/data/twitter/json/year=2017/month=04/day=01'`. The smaller the table the faster the queries because we only need to read the files that contain the information we are looking for. This approach is however incovenient because a new table must be created whenever we want to query a new range of dates.
#
# A better approach consists of partitioning the table using Hive's `PARTITIONED BY` table properties, for instance `PARTITIONED BY(year STRING, month STRING, day STRING)`. Note that in this example, `year`, `month` and `string` are not columns of the HDFS files used to create the tables. Rather, they indicate folders of the file hierarchy under which the files are organized.
#
# After you create a partitioned external table, the next step is to the add each partition to the table individually, using the DDL commands such as `ALTER TABLE ... ADD PARTITION(year='2017',month='01',day='01') LOCATION '/data/twitter/json/year=2017/month=01/day=01' ...`. This is obviously a cumbersome approach if we have many partitions. Hence the particular naming convention used in this example `year=.../month=.../day=...`, which is used in Hive to declare the partitions, so that they can automatically be created using the [MSCK REPAIR TABLE](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-RecoverPartitions(MSCKREPAIRTABLE)) command. 

# %%
##
query = """DROP TABLE IF EXISTS {0}.twitter""".format(username);
cur.execute(query)

# %%
query = """
CREATE EXTERNAL TABLE {0}.twitter(json STRING)
  PARTITIONED BY (year STRING, month STRING, day STRING)
  STORED AS TEXTFILE
  LOCATION '/data/twitter/json/'
""".format(username)
cur.execute(query)

# %%
query = """MSCK REPAIR TABLE {0}.twitter""".format(username);
cur.execute(query)

# %% [markdown]
# After the table `twitter` is created, select one row with a select command (`LIMIT 1`). You can specify a particular day using the standard SQL `WHERE` clause to filter the result. In this case `WHERE year=... AND month=... AND day=...`. Use the output of the select query to identify the json fields where the the language and the timestamp information of the tweet are stored. You can use http://jsonprettyprint.com/, or the command in [exercises-1_cli (B.)](exercises-1_cli.md) to pretty print the json string.

# %%
##
query = """
SELECT * FROM {0}.twitter WHERE year=2020 AND month=12 AND day=31 LIMIT 1
""".format(username)
cur.execute(query)
cur.fetchall()

# %% [markdown]
# Compare to the other method:

# %%
query = """
DROP TABLE IF EXISTS {0}.twitter_2020_12_31
""".format(username)
cur.execute(query)

# %%
query = """
CREATE EXTERNAL TABLE {0}.twitter_2020_12_31(json STRING)
  STORED AS TEXTFILE
  LOCATION '/data/twitter/json/year=2020/month=12/day=31'
""".format(username)
cur.execute(query)

# %%
query = """
SELECT * FROM {0}.twitter_2020_12_31 limit 1
""".format(username)
cur.execute(query)
cur.fetchall()

# %% [markdown]
# ----
# **Step 2.** Extract JSON fields from raw text format.
#
# Hive parses the file as raw text format, ignoring its JSON structure.
#
# In the next query we use the following [User Defined Functions](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF) to extract and process the JSON fields from the text.
#
# * get_json_object
# * from_unix_time
# * cast
# * min
# * max
#
# Can you guess the meaning of this Hive command?
#
# For further reading, you can also learn more about the subtle distinction between [**order by** and **sort by**](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+SortBy). It describes the side effects of the underlying _MapReduce_ technology on top of which Hive is built.

# %%
query = """
WITH q AS (
    SELECT
        get_json_object(json, '$.lang') AS lang,
        from_unixtime(
            cast(
                cast(
                    get_json_object(json, '$.timestamp_ms') as bigint
                ) /1000 as bigint
            )
        ) AS time_str
    FROM {0}.twitter WHERE year=2020 AND month=12 AND day=31
)
SELECT lang,count(*) as count,min(time_str) AS first_ts_UTC,max(time_str) AS last_ts_UTC
FROM q
GROUP BY lang
ORDER BY count desc
""".format(username)
pd.read_sql(query,conn)

# %% [markdown]
# ----
# **Step 3.** Use available Serialization/Deserialization (_SerDe_) libraries.
#
# Using `get_json_object` in every `select` queries can be cubersome, and error prone. Hive provides the [_SerDe_ framework](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-RowFormats&SerDe) to simplify the data IO serialization and deserialization. _SerDe_ properties are specified when Hive tables are created.
#

# %%
query="""DROP TABLE IF EXISTS {0}.twitter_serde""".format(username)
cur.execute(query)

# %%
query="""
CREATE EXTERNAL TABLE {0}.twitter_serde(
        timestamp_ms STRING,
        lang STRING
    )
    PARTITIONED BY (year STRING, month STRING, day STRING)
    -- ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.JsonSerDe'
    ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
    WITH SERDEPROPERTIES(
        "ignore.malformed.json"="true"
    )
    STORED AS TEXTFILE
    LOCATION '/data/twitter/json/'
""".format(username)
cur.execute(query)

# %%
query = """MSCK REPAIR TABLE {0}.twitter_serde""".format(username);
cur.execute(query)

# %%
query="""
    SELECT * FROM {0}.twitter_serde WHERE year=2016 AND month=05 AND day=15 LIMIT 10
""".format(username)
pd.read_sql(query,conn)

# %% [markdown]
# -----
# ## Part III - Hive over HBase
#
# Hive is a data warehouse built on top of Hadoop. Hive queries get translated into MapReduce batch jobs that run on a distributed cluster. Hive is thus better for bulk inserts or updates of millions of rows at a time. It is not designed for fast individual lookups, operations on individual rows, or for real-time data. Other databases, such as MongoDB, Cassandra, [Apache Accumulo](https://accumulo.apache.org/) or [HBase](https://hbase.apache.org/) are better suited for real-time data. However they lack the relational DBMS flavor of Hive and an SQL-like interface. They are not ideal for complex relational queries.
#
# In the following exercises, we will illustrate how to get the best from both worlds by integrating Hive with HBase. With this configuration it is possible to ingest high rates of individual rows of data, such as sensor measurements, in HBase, and run batch queries, possibly mixed with data HDFS data, using Hive.
#
# More details about what follows can be found in the [Hive/HBase integration](https://cwiki.apache.org/confluence/display/Hive/HBaseintegration) documentation.

# %% [markdown]
# -----
# ### HBase
#
# HBase in a few bullet points:
# * HBase is a _noSQL_ (or non-relational) database.
#   - You use HBase when you need random, realtime read/write access to Big Data.
#   - HBase is schema-less, and is not relational DBMS. We do not use it for relational queries.
# * It is a _key-value_ store - rows in HBase are indexed by their row keys.
# * It is a _wide column store_. It can handle billions of rows on millions of columns on clusters of commodity hardware.
#   - Columns are organized into _column families_.
#   - _Column families_ must be conjured upfront. They apply to all the rows.
#   - Columns and their names are not fixed. They are conjured on the fly when rows are created or updated.
#   - It is a _sparse_ database. Empty columns take no space, they do not exist in HBase.
#   - Column values are versioned.
# * Tables have a _name_ and a _namespace_, are are uniquely identified by _namespace:name_
# * Main data model operations on HBase are:
#   - `put`: add a new row, or update an existing row
#   - `get`: get the value from a row
#   - `scan`: iterate over range of contigous rows, optionally with a _Filter_.
#   - `delete`: delete a row
# * It is built on top of HDFS. This is counter intutive, given HDFS's block-based nature. HBase manages it with periodic data compaction.
#
#
# Conceptually, the structure of a HBase table looks like this:
#
# | Row key       | timestamp | ColumnFamily1 | ColumnFamily2  |
# | ------------- |:---------:|:---------------:|:----------------:|
# | 01.12.2019/80:06____:17004:000 | 1583867978 | produkt_id=Zug,linenid=17004 | bpuic=8500090 |
# | 01.12.2019/80:06____:17017:000 | 1583868321 | produkt_id=Bus,linenid=701   | bpuic=8301093 |
#
# A _{row,column-family:column,version}_ specifes a _cell_ in the table.
#
# References: [hbase.apache.org](https://hbase.apache.org/book.html)
#

# %% [markdown]
# ----
# #### Create a connection to HBase
#
# We use [happybase](https://happybase.readthedocs.io/en/latest/) to connect remotely to the HBase server. The happybase API supports a very limited subset of all the commands possible with HBase. For more complex tasks we would use the `hbase` command line interface.
#
# The python package is already installed. Otherwise, it can be installed with `pip install happybase`

# %%
import happybase
hbaseaddr = os.environ['HBASE_SERVER']
hbaseaddr = "iccluster029.iccluster.epfl.ch"
hbase_connection = happybase.Connection(hbaseaddr, transport='framed',protocol='compact')

# %% [markdown]
# ----
# #### Create an HBase table
#
# We use your ID (from variable username) for the table namespace in HBase, as we did in Hive.
#
# You may need to delete the table first if it exists. HBase table must be `disabled` before they can be deleted or altered.

# %%
try:
    hbase_connection.delete_table('{0}:sbb_hbase'.format(username),disable=True)
except Exception as e:
    print(e.message)
    pass

# %% [markdown]
# -----
# Create a new HBase table, called `sbb_hbase` under your namespace.
#
# Note: if we do not create the table, a default table will be created when we create the Hive table.
#
# * The table has two column families.
# * Values in the first column families are versioned, we keep the last 10 values.

# %%
hbase_connection.create_table(
    '{0}:sbb_hbase'.format(username),
    {'cf1': dict(max_versions=10),
     'cf2': dict()
    }
)

# %% [markdown]
# -----
# List all the tables in HBase

# %%
print(hbase_connection.tables())

# %% [markdown]
# -----
# Inspect the tables. The properties of the column families can be specified when the table is created, e.g. `{ 'cf1': dict(max_version=10,block_cache_enabled=False) }`

# %%
hbase_connection.table('{0}:sbb_hbase'.format(username)).families()

# %% [markdown]
# -----
# We can scan the HBase table to verify that it is empty

# %%
for r in hbase_connection.table('{0}:sbb_hbase'.format(username)).scan():
    print(r)

# %% [markdown]
# -----
# Create an external Hive table on top of the HBase table. We first delete it if it exists.
#
# The table will contain the following fields:
# * `RowKey`
# * `BETRIEBSTAG`
# * `FAHRT_BEZEICHNER`
# * `ABFAHRTSZEIT`
# * `BPUIC`
#
# RowKey is the key we will use to index the row. Other columns are populated from the sbb table.
#
# Note:
# * The table is stored using the `org.apache.hadoop.hive.hbase.HBaseStorageHandler` for HBase.
# * TBLPROPERTIES `hbase.table.name` specifies the name of the HBase table that the external Hive table should point to. It is optional, and default to the same name as the Hive table.
# * SERDEPROPERTIES `hbase.columns.mapping` defines the mapping between the Hive columns and the HBase columns. The definition are listed in the Hive column order, that is `RowKey` maps to `:key`, the key of the HBase table, `BETRIEBSTAG` maps to `cf1:betriebstag` in HBase (Column family cf1, column betriebstag), and so on.
#

# %%
query = """
DROP TABLE {0}.sbb_hive_on_hbase
""".format(username)
cur.execute(query)

# %%
query = """
CREATE EXTERNAL TABLE {0}.sbb_hive_on_hbase(
    RowKey string,
    betriebstag STRING,
    fahrt_bezeichner STRING,
    abfahrtszeit STRING,
    bpuic BIGINT
) 
STORED BY 'org.apache.hadoop.hive.hbase.HBaseStorageHandler'
WITH SERDEPROPERTIES (
    "hbase.columns.mapping"=":key,cf1:betriebstag,cf1:fahrt_bezeichner,cf2:abfahrtszeit,cf2:bpuic"
)
TBLPROPERTIES(
    "hbase.table.name"="{0}:sbb_hbase",
    "hbase.mapred.output.outputtable"="{0}:sbb_hbase"
)
""".format(username)
cur.execute(query)

# %%
query = """
SHOW TABLES FROM {0}
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# -----
# The external Hive table is backed by the HBase table, which is currently empty.

# %%
query = """
SELECT * FROM {0}.sbb_hive_on_hbase LIMIT 1
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# -----
# We may now populate the HBase table with SBB data. We do this through a `insert overwrite ... select`, as before. We copy from the `{0}.sbb_orc_2020_12` created in an earlier exercise, to `{0}.sbb_hive_on_hbase`. This command may take a few minutes to complete.

# %%
query="""
INSERT OVERWRITE TABLE {0}.sbb_hive_on_hbase
    select
         concat(BETRIEBSTAG,":",FAHRT_BEZEICHNER) as RowKey,
         betriebstag,
         fahrt_bezeichner,
         abfahrtszeit,
         bpuic
    FROM {0}.sbb_orc_2020_12 limit 20
""".format(username)
cur.execute(query)

# %% [markdown]
# -----
# A scan the HBase table shows the rows inserted by Hive

# %%
for r in hbase_connection.table('{0}:sbb_hbase'.format(username)).scan():
    print(r)

# %% [markdown]
# ------
# Cleanup (optional)

# %%
query = """
DROP TABLE {0}.sbb_hive_on_hbase
""".format(username)
cur.execute(query)

# %%
query = """
DROP TABLE {0}.sbb_orc_2020_12
""".format(username)
cur.execute(query)

# %%
query = """
DROP TABLE {0}.sbb_csv_2020_12 
""".format(username)
cur.execute(query)

# %%
try:
    hbase_connection.delete_table('{0}:sbb_hbase'.format(username),disable=True)
except Exception as e:
    pass

# %% [markdown]
# #### Resources:
# * Hive toturial: [https://cwiki.apache.org/confluence/display/Hive/Tutorial](https://cwiki.apache.org/confluence/display/Hive/Tutorial)
# * HBase integration: [https://cwiki.apache.org/confluence/display/Hive/HBaseintegration](https://cwiki.apache.org/confluence/display/Hive/HBaseintegration)

# %%
