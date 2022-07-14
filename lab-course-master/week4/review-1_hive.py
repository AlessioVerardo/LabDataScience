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
# # Data Wrangling with Hadoop Distributed File Systems and Hive
#
# This tutorial will teach you the basic steps of building your first data lake.
#
# You will start by copying a dataset from an external source into the Hadoop distributed file system, then learn how to query that data using the Hive Query Language (HiveQL).
#
# You will also use Hive to prepare your data for operations in a large data lake and convert it to a more efficient format for querying.
#
# This is the first part of a two-parts revision notebooks about Hive. The notebooks are designed so that they can be reviewed as standalone materials.

# %% [markdown]
# ---
# ### A small reminder about your work environment
#
# You access this page from your personal laptop. During the next set of exercises, no data will be copied to your laptop, and no processes will be executed on it (other than your web browser).
#
# Instead, as shown in figure 1, all the action will take place in your Jupyterlab environment running on our servers at EPFL (in the middle of the figure). This environment is equivalent to a small personal computer on which you can run applications, like this Jupyter notebook, via a web interface. An individual Jupyterlab environment is created for each of you when you _Create a new interactive environment_ in RENKU. You can close your web browser (and shut down your laptop), and still be able to log back into that environment later. Note, however, that in order to conserve our computing resources, the environment will shut itself down if you do not reopen it within 24 hours.
#
# Your Jupyterlab environment is very lightweight and has only a fraction of the CPU and limited RAM, which is obviously not enough to handle the kind of terabyte data operations we are about to discuss. Instead, all data and heavy operations will be stored and executed on a Hadoop big data software stack on a separate computer cluster (shown on the right of the figure).
#
# In summary, the python program in this notebook runs in a Jupyterlab environment outside of your laptop, on the EPFL servers. You interact with this notebook via a web interface. Some commands are executed on the big data backend and the results are sent back to the notebook where you can further analyze and display them. In the remainder of this discussion, we will refer to this notebook as your _local_ environment. Even if it is not running on your laptop, that is how it appears to you. We will call the big data cluster the _remote_ servers.
#
# ![work environment](./figs/work-env.svg)
#
# Fig. 1 - overview of your work environment

# %% [markdown]
# ---
# ### About the data
#
# In this part, you will leverage Hive to prepare a data set for exploratory analysis.
#
# The data is from SBB/CFF/FFS Data and is published by the [Open Data Platform Swiss Public Transport](https://opentransportdata.swiss).
#
# In this particular example you will play with the **Ist-daten** (real data), available from <https://opentransportdata.swiss/en/dataset/istdaten>.
#
# More information about this data will be provided in the notebook [review-2_hive.py](./review-2_hive.py) which you will see next, or the [opentransportdata cookbook](https://opentransportdata.swiss/en/cookbook/actual-data/)
#
# As a preview, this is the table _schema_ of the data that you will be using:
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
# - `ANKUNFTSZEIT`: arrival time at the stop according to time tables
# - `AN_PROGNOSE`: actual arrival time
# - `AN_PROGNOSE_STATUS`: show how the actual arrival time is calcluated
# - `ABFAHRTSZEIT`: departure time at the stop according to time tables
# - `AB_PROGNOSE`: actual departure time
# - `AB_PROGNOSE_STATUS`: show how the actual departure time is calcluated
# - `DURCHFAHRT_TF`: boolean, true if the transport does not stop there
#
#

# %% [markdown]
# ---
# #### 1. Notebook Initialization
#
# Think of this notebook as a python program that will be accessing a remote _Hive_ data warehouse. You need to tell this program the IP address of Hive. You must also identify yourself to the Hive so that it knows who you are. Because each of us is uniquely identified, this information cannot be hardcoded into this notebook; it must be indicated at runtime.
#
# For your convenience, the address of the Hive data warehouse and your user name are made available via _environment variables_. The environment variables are set dynamically using values taken from your personal profile when the interactive environment is started. They live outside of this notebook, but you can get their values from `os.environ['VARIABLE_NAME']`. We find it more convenient to copy their values into python variables and call them respectively `hiveaddr` and `username` for short.
#
# Note that this is a non-standard method to connect to Hive on the big data cluster. We added a bit of black magic to make the task easier for you. In other computing environments they may do things differently, but the principle remains the same.
#
#
# <div class="alert alert-info" style="margin: 10px">
# <strong>When In Python</strong>
# <ul><li>We are importing the python package <strong>os</strong>. Python comes with a large number of standard packages preinstalled. Many more contributed packages can be found online and installed using a python package manager such as <strong>conda</strong> or <strong>pip</strong>. A package contains variable and function declarations from a group of related operations (such as trigonometric functions and the value of pi). A python program starts with the minimum packages it needs to perform the most basic tasks. This is to keep its memory footprint light and loading time short. If you need to use functions from other package installed on your machine, they must be imported manually into your program using the <strong>import</strong> command.
#     <li>The <strong>os</strong> package defines the variable <strong>environ</strong>. It is a dictionary that contains the names and corresponding values of all the environment variables. You can do <code style="color:black;">print(os.environ)</code> to display all of them.</li>. In this case we are only interested in <code style="color:black;">HIVE_SERVER2</code> and <code style="color:black;">RENKU_USERNAME</code>
# </ul>
# </div>

# %%
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

hiveaddr = os.environ['HIVE_SERVER2']
username = os.environ['RENKU_USERNAME']
print(hiveaddr)
print(username)

# %% [markdown]
# ---
# #### 2. Copy data into HDFS
#
# **Note:** you do not need to do the exercises of section **2.** if you have already completed the HDFS exercises of [week 3](../week3/exercises-1_hdfs.md).
#
# You've got your hands on terrabytes of data, ... now what?
#
# Maybe this data is somewhere on a portable drive. You must put this data in a place where you, and your collaborators can easily retrieve it, and wrangle, munge, and study it with the big data machinery. This is your first step in building a _Data Lake_.
#
# Among the many possible options you should consider using the _Hadoop File Distributed Systems_ (HDFS in short). The advantages offered by HDFS are numerous, among which you should remember that it is:
#
# * **Cost-effective** - it can be built with commodity hardware using low-cost disk storage.
# * **Scalable** - add more computers with storage and increase your total storage capacity. Maximum file size is limited by the total size of the storage, not the size of individual storage devices.
# * **Distributed** - files are accessible from anywhere on the network via a simple interface.
# * **High availability** - storage redundancies ensure that if one drive fails, a copy of your files can be found elsewhere.
# * **Efficient** - HDFS file storage is organized to maximize parallel read/write capabilities and optimize network operations.
#
#
# The HDFS command interface enables the lowest level of file storage operations that you, as a data scientist, will need to know about when interacting with the big data software stack (There are other similar types, such as GPFS-SNC/FPO, or the S3 object store, but we won't study them here.) This is because in HDFS, files are more efficiently written or read in one (very) large chunk at a time. So you use the HDFS command mainly to store, move or read very large, unstructured files in their entirety. It's not something you want to use for instance to randomly query or access a few bytes in a file. For that, you would prefer to use higher-level commands provided by data management systems, such as Hive and HBase, which implements an optimization layer on top of HDFS to speed up searching and reading or writing any size of data.

# %% [markdown]
# **Hand-on with HDFS**
#
# In this simple example, we copy one day of SBB data (07.04.2021) to the HDFS storage on the _remote_ servers. You can find the data you want to copy from SBB's [open transport data](https://opentransportdata.swiss/en/dataset/istdaten) under the name `2022-02-27_istdaten.csv`. Use `curl` or `wget` in a Terminal to download it in the `/tmp/` folder. Note: this file is several hundreds MBytes, do not try to open it in this local environment.
#
# The steps are as follow:
# * Connect to the remote HDFS server
# * Create a destination folder for the data on HDFS
# * Copy the local data to the destination folder

# %% [markdown]
# ##### 2.1 Create a connection to the remote HDFS server
#
# As mentioned before, files in HDFS are accessed over the network. We create a python object to natively use HDFS from the local Python environment. There exist several solutions for this task. In this example we will use `HDFileSystem` from the [hdfs3](https://hdfs3.readthedocs.io/en/latest/) python package. It is already installed in your local interactive environment.
#
# Your environment is preconfigured with the address of the HDFS server, and automatically connects under your username.

# %%
from hdfs3 import HDFileSystem
hdfs = HDFileSystem()

# %% [markdown]
# You may now use this `hdfs` object to interact with the remote HDFS server.
#
# First, list the file content of your HDFS home directory using the [HDFileSystem.ls(path)](https://hdfs3.readthedocs.io/en/latest/api.html#hdfs3.core.HDFileSystem.ls) command. Each of you should have a home folder. You can access your HDFS folder using `.` (dot) as a shorthand for the path.
#
# In linux the _dot_ notation refers to the currently active folder. In HDFS there is no concept of active folder, and `.` is always your home folder.
#
# Note that in the output, the absolute path of your home folder is `/user/<username>`. Like in linux, `/` is the root of the folder arborescence, and path names beginning in `/` are relative to the root. Otherwise they are relative to `.`, which is always your home folder.

# %%
hdfs.ls(".")

# %% [markdown]
# ##### 2.2 Create a destination folder for the data
#
# Use the [HDFileSystem.makedirs(path,mode)](https://hdfs3.readthedocs.io/en/latest/api.html#hdfs3.core.HDFileSystem.makedirs) command to create a destination folder in your home directory. Give the name `work1` to this folder. Then use the `ls` command to verify that the folder has been created in your home HDFS folder.

# %%
hdfs.makedirs("./work1")
hdfs.ls(".")

# %% [markdown]
# ##### 2.3 Copy the data to the remote HDFS server
#
# Use the [HDFileSystem.put(filename,path)](https://hdfs3.readthedocs.io/en/latest/api.html#hdfs3.core.HDFileSystem.put) command to copy the local data filename `2021-04-07_istdaten.csv` to HDFS under `./work1/2021-04-07_istdaten.csv` in your home folder. Then verify that the file has been copied (note the file must have been copied locally to /tmp).

# %%
hdfs.put("/tmp/2022-02-27_istdaten.csv","./work1/2022-02-27_istdaten.csv")
hdfs.ls("./work1")

# %% [markdown]
# This is all we need to know about HDFS for this exercise. We encourage you to review the [online documentation](https://hdfs3.readthedocs.io/en/latest/api.html#) for a list of other HDFS operations available via the Python HDFileSystem interface.
#
# Note that this interface only provides a subset of all the possible HDFS commands. The most courageous among you are invited to experiment with the command line interface `hdfs`, also available in this environment: open a _Terminal_ or a _Bash_ notebook from the _Jupyterlab Launcher_, and type `hdfs dfs` on the command prompt. It will return a list of possible commands. See the HDFS [file systems shell](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/FileSystemShell.html) command line documentation for more details.

# %% [markdown]
# ---
# #### 3. Query the Data with Hive
#
# In the following suite of exercises you are about to familiarize yourself with the Hive Query Language (HiveQL).
#
# You will first put the sbb data in Hive table to explore it. You will then use Hive to prepare your data for operations in a large data lake and convert it to a more efficient format for querying.
#
# The steps are as follow:
# * Connect to Hive
# * Create a Hive Database
# * Declare (create) a Hive Table using the data you just copied on HDFS
# * Query the Table using Hive Query Language
# * Create a Hive Table using the same data, and a more efficient data format
#
# As you go along, you will be using three categories of Hive commands:
# * Data Definition Language ([DDL](https://cwiki.apache.org/confluence/display/Hive/LanguageManual%20DDL)) commands - to create and modify Hive objects such as databases and tables.
# * Data Modification Language ([DML](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DML)) commands - to add, delete or update data in Hive table.
# * Data [retrieval](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Select) commands - to query the content of Hive tables.
#
# 📝 The SQL language is case insensitive. In the following we will apply the generally accepted SQL convention, and use upper case for SQL keywords, and lower case for user-defined table and column names.

# %% [markdown]
# ##### 3.1 Connect to Hive
#
# Like HDFS, the Hive data warehouse runs on a remote server outside this python notebook. Before interacting with Hive you must establish up a connection with the server. We use `hive` from the [pyhive](https://pypi.org/project/PyHive/) Python package (and like before with HDFS there are many other packages available).
#
# We create two python objects:
#
# * A hive connection `conn` - returned by `hive.connect(host,port,username)`, this python object is the contact point (programming interface) between your notebook and the remote Hive server. The object constructor needs the details of the hive server address and your username, previously defined in _1. Notebook Initialization_.
# * A hive `cursor` - returned by `conn.cursor()`, you will use this object to execute Hive queries with `conn.execute("query")`, and iterate over the result sets with `conn.fetchall()`.

# %%
from pyhive import hive

(hivehost,hiveport)=hiveaddr.split(':')
conn = hive.connect(host=hivehost,port=hiveport,username=username)
cursor = conn.cursor()

# %% [markdown]
# ##### 3.1 Create your database
#
# A Hive databases is a namespace or a collection of tables. Having each your own database ensures that all your tables are grouped together, and users of other databases can reuse the table names without creating conflicts with yours.
#
# You do not want to create the database every time you rerun this notebook. Therefore you create the database only if it does not already exist with the `CREATE DATABASE IF NOT EXISTS` command. The name of the database can be anything, but we recommend that you name it with your previously defined _username_, so that it can be easily identified.
#
# In addition, you will also make this database the default with the `USE` command. Otherwise, you should specify the database name in all your Hive operations, such as `SHOW TABLES IN databasename`, and `SELECT * FROM databasename.tablename` so that there are no ambiguities.
#
# See the Hive documentation for more details about the [CREATE DATABASE](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-Create/Drop/Alter/UseDatabase) command.
#
# <div class="alert alert-info" style="margin: 10px">
# <strong>When In Python</strong><p>
# <div>
# We will often use the "{0}".format(list) Python construct. It is used to place a list of values at specific positions in a string. The numbers inside the curly brackets are the positions in the list (starting at 0) of the values that you want to insert at the bracket's position.
# <p><p>
# For instance:
#
# <code style="color:black;">print("The {1} jumps over the {0}".format('lazy dog','quick brown fox'))</code>
#     </div>
# </div>

# %%
query = "CREATE DATABASE IF NOT EXISTS {0}".format(username)
cursor.execute(query)

# %%
query = "USE {0}".format(username)
cursor.execute(query)

# %% [markdown]
# ##### 3.2 Create a table
#
# Before proceeding, you should know that in Hive, a table is made of two things:
# * **Data** - this is the actual content of the table
# * **Metadata** - or _data about the data_, are the properties of data, such as column names and their types (i.e. their schema), which are not part of the data but are needed by Hive in order to be able to process that data. The metadata is created when you create the Hive table. Note that the same data can be given different column names and types, and described in different ways, each resulting into a different metadata.
#
# Also Hive knows two different types of tables: **Managed** and **External**.
#
# * **Managed tables**: The fundamental difference with external tables is that _Hive assumes that it owns the data_ for managed tables. That means that the data, its properties and data layout will and can only be changed via Hive commands. The data still lives in a normal file system and nothing is stopping you from changing it without telling Hive about it. If you do though it violates Hive invariants and you might see undefined behavior. Another consequence is that both data and metadata of managed tables is attached to Hive. So, whenever you change a table (e.g. drop a table or a partition) the data is also changed accordingly (in this case the data is deleted). This is very much like with traditional RDBMS where you would also not manage the data files on your own but use a SQL-based access to do so. [DROP TABLE](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-DropTable) (the opposite of CREATE TABLE) deletes data for managed tables.
#
# * **External tables**: For external tables Hive assumes that it only manage the table metadata, and it does not manage the data, i.e. content of the table, itself. An external table describes the metadata or schema about data files that were created beforehand. External tables can access data stored in sources such as remote HDFS locations. External table files can be accessed and managed via other methods outside of Hive. If the structure or partitioning of an external table is changed outside Hive by such a process, a Hive [ALTER TABLE](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-AlterTable) statement can be used to refresh its metadata information. [DROP TABLE](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-DropTable) only deletes metadata for external tables, the data itslef is not deleted.
#
# <div class="alert alert-block alert-warning" style="margin: 10px">
# <strong>When In Hive</strong><p>
# <div>
#     ⚠️ You absolutely do not want to lose your external files when you drop the table.
#     Always use <strong>external</strong> tables when files are already present or in remote locations, and the files should remain even if the table is dropped.
# </div>
# </div>

# %% [markdown]
# In Hive you cannot create two tables with the same name in the same database. You could conditionally create the table with a `CREATE TABLE ... IF NOT EXISTS` statement as you did for the database, but if you do so, you run the risk of using a pre-existing table that has schema different than the one you originally intended.
#
# Therefore, it is better to unconditionally delete the table with the [DROP TABLE IF EXISTS](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-DropTable)  statement before creating a new table. And this is without consequences, because you will only create _external_ tables of that name, and as we saw earlier, this operation only deletes the metadata from _external_ tables, which take a fraction of a second to recreate.

# %%
cursor.execute('DROP TABLE IF EXISTS my_sbb_istdaten_csv')

# %% [markdown]
# You can now create the table. Since you want to use the data previously uploaded to HDFS, you need to use the [CREATE EXTERNAL TABLE](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-ExternalTables) command, in order to create an _external_ table on an existing HDFS data source.
#
# This command takes the input file and makes it appear as a Hive table.
#
# The anatomy of the command is:
# * `create external table table_name` - create an externa table.
# * `(column_name column_type, ...)` - declare the name and type of each column (schema)
# * `row format delimited fields termnated by ';'` - this is a comma (semicolon) separated value file of fixed number of fields per row, separated by `;`.
# * `stored as textfile` - the external data is stored in simple text files
# * `location '/user/username/istdaten'` - this is the destination folder in your HDFS home were you copied the file ({0} is replaced by your username)
# * `tblproperties ('skip.header.line.count'='1', ...)` - other table properties. In this example we are ignoring the first line of each file used in the input. They contain the column headers and you do not want them to appear as normal rows in the table.

# %%
query = """
CREATE EXTERNAL TABLE my_sbb_istdaten_csv(
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
     LOCATION '/user/{0}/work1'
     TBLPROPERTIES ('skip.header.line.count'='1','immutable'='true')
""".format(username)
cursor.execute(query)

# %% [markdown]
# You can now list all the tables in your database with the Hive [SHOW TABLE](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-ShowDatabases) command, and verify that the new table was created. Since the query returns a result (the list of tables in your database), you want to use the `cursor.fetchall()` command to iterate over them.

# %%
cursor.execute('SHOW TABLES')
cursor.fetchall()

# %% [markdown]
# ##### 3.3 Query the table
#
# The data is now available as Hive data and can be queried with the Hive language, which is very similar to SQL and consists of [SELECT](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Select) statements. Note that creating the external table did not move or copy any data. All you did was tell Hive how to interpret the data you copied to the _istdaten_ folder in your home HDFS folder.

# %%
cursor.execute('SELECT * FROM my_sbb_istdaten_csv LIMIT 2')
cursor.fetchall()

# %% [markdown]
# The above command iterates over the results set and print the output in a raw format. We recommend that you use the [read_sql](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html) function from the **pandas** Python package to pretty print the results of your queries.
#
# Pandas is an important part of the Data Scientist's toolbox and we strongly encourage you to consult the
# [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide).  Pandas DataFrames provides a simple interface to operate on tabulated data. Pandas DataFrames are not meant to handle big data however, and we will only use them sparingly in this set of exercises.
#
# Returning to `pandas.read_sql(query, connector)`, this command uses the Hive connector to create a temporary cursor, and uses it to query the table and display the results in a well-formatted output. We limit the number of rows to 5, otherwise the resulting output would be too large for this notebook.

# %%
import pandas as pd
pd.read_sql('SELECT * FROM my_sbb_istdaten_csv LIMIT 5', conn)

# %% [markdown]
# ##### 3.4 Copy the table into a new table using a more efficient format
#
# You just created a Hive table out of an external input file stored on HDFS. Note that you specified the folder `istdaten` as the location of the file, and if you add more input files to this folder, the content of the files will automatically become part of the table in your next query - as long as they are text files consisting of `;`-separated fields, and they can all be coerced into the schema you used to create the table. Compress the files, or organize them in sub-folders, and Hive will know how to handle it.
#
# The text format is most commonly encountered when sharing data, because it is universal and human readable. This format is however not recommended for storing huge data sets in a Data Lake because it is very inefficient to search and process. Instead, it is recommended to use the **ORC** or **PARQUET** format for this purpose.

# %% [markdown]
# Converting an input data into **ORC** or **PARQUET** is easily done by creating new table in the desired format. In the following example we use [CREATE TABLE](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable) to create a Hive **managed** table, stored in **ORC** format. Since we want to create a copy of the original table (in a different format), we use the same schema as before. Most notable differences are:
# * `create table(...)` - create a Hive-managed table (it is not external)
# * `store as orc` - the data is in ORC format
# * `location '/user/{0}/orc/istdaten'` - the content of the table is stored in your home HDFS folder under _./orc/istdaten_ (the folder is created if needed)
# * `tblproperties ('orc.compress'='SNAPPY',...)` - the data is compressed with SNAPPY, an HDFS-friendly compression algorithms that provides a relatively good CPU to compression rate tradeoff.

# %%
cursor.execute('DROP TABLE IF EXISTS my_sbb_istdaten_orc')

# %%
query="""
CREATE EXTERNAL TABLE my_sbb_istdaten_orc(
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
    STORED AS ORC
    LOCATION '/user/{0}/orc/istdaten'
    TBLPROPERTIES ('orc.compress'='SNAPPY','immutable'='true')""".format(username)
cursor.execute(query)

# %% [markdown]
# The table you just created is empty. You must now [INSERT](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DML#LanguageManualDML-InsertingdataintoHiveTablesfromqueries) into it the output of a `SELECT` command from the table that you want to convert. There are two forms of insert. `INSERT OVERWRITE` will overwrite any existing data in the table. `INSERT INTO TABLE` will append to the table, keeping the existing data intact.
#
# The above method is known as CREATE TABLE and INSERT. Tables can also be created and populated by the results of a query in one Create-Table-As-Select [(CTAS)](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTableAsSelect(CTAS)) statement. The advantage of CTAS is that the table created by CTAS is atomic, meaning that the table is not seen by other users until all the query results are populated. So other users will either see the table with the complete results of the query or will not see the table at all. 

# %%
query="""
INSERT OVERWRITE TABLE my_sbb_istdaten_orc SELECT * from my_sbb_istdaten_csv
"""
cursor.execute(query)

# %% [markdown]
# You may now query the new table:

# %%
pd.read_sql('SELECT * from my_sbb_istdaten_orc LIMIT 5', conn)

# %% [markdown]
# You can also verify that a new folder (`./orc/istdaten`) has been created in your HDFS home folder. The table is Hive **managed**, and dropping the table deletes the folder and all its content.

# %%
hdfs.ls("./orc")

# %%
