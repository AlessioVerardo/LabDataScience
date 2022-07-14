# %% [markdown]
# # Fist Steps with HIVE
#
# In the next exercises you will learn basic Hive commands used to manage and query large data sets stored on HDFS.
#
# ----
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
import re
import os
import numpy as np
import pandas as pd
from pyhive import hive
from scipy.stats import entropy

# Set python variables from environment variables
username  = os.environ['RENKU_USERNAME']
hive_host = os.environ['HIVE_SERVER2'].split(':')[0]
hive_port = os.environ['HIVE_SERVER2'].split(':')[1]

# create connection

conn = hive.connect(
    host=hive_host,
    port=hive_port,
    username=username
)

# create cursor
cur = conn.cursor()

# %% [markdown]
# ## Data from the Eukaryotic Promoter Database
#
# Data ftp source: ftp://ccg.epfl.ch/epdnew/H_sapiens/006/db
#
# In this part, we use Hive to explore human promoter DNA sequences and associated gene expressions in different tissues. We use data sourced from the [Eukaryotic Promoter Database (EPD)](https://epd.epfl.ch/). The files are made available by the SIB under a [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
#
# > Background reminder: Promoters are regulatory DNA sequences located just before the start of genes. These sequences contain motifs that affect the activity (expression) of genes. The effect of a promoter on its gene is tissue-dependent: the same promoter sequence can increase gene expression in the brain, but not in the liver.
#
#
# You can find the data on HDFS at the path `/data/epd/`
#
# There are 3 tables with the following schemes. The full description of features can be found at this ftp address: ftp://ccg.epfl.ch/epdnew/README
#
#
# `promoter_sequences`:
#
# * `promoter_name`: Unique identifier for each promoter sequence.
# * `sequence`: Fixed-length DNA sequence of the promoter, truncated to bases -49 to +10 relative to annotated TSS.
#
# `promoter_expression`
#
# * `promoter_name`: Unique identifier for each promoter sequence.
# * `expression`: Gene expression associated with this promoter. Number of CAGE tags across samples in a 100bp window around TSS.
# * `position_vs_TSS`: Sample-specific TSS position detected relative to annotated TSS position.
# * `sample`: The name of the sample in which expression is measure.
#
# `promoter_samples_expression`:
#
# * `promoter_name`: Unique identifier for each promoter sequence.
# * `num_active_samples`: The number of samples with detectable activity for this promoter.
# * `average_expression`: Average gene expression. Mean number of CAGE tags across samples in a 100bp window around annotated TSS.

# %% [markdown]
# ## First step with Hive
#
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
# We use the `location` property to instruct Hive to create the our database in the HDFS folder we created at the end of our previous set of exercises, under /user/_**username**_/hive.

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
# **Q1**: Create an **external** Hive table for `promoter_expression`. Hive will create a reference to the files located under `/data/epd/csv/promoter_expression/` on the HDFS storage, and apply a table schema on it, but it will not manage the file itself if the table is declared **external**. When you drop an external table, only its definition in Hive is deleted, the content of `promoter_expression` is preserved. If on the other hand you forget to declare the table **external**, the files under the HDFS folder `promoter_expression` will be deleted (for all of us) when you drop the table.
#
# Feel free to browse the content of `/data/epd/csv/` using the `hdfs dfs -ls` command in a Terminal.
#
# You will need to change one line in the code below in order to get it to work.
#
# See the [Hive DDL](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL) reference manual.

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>WARNING! </b> As a rule of thumb - if you create Hive tables from existing HDFS files, <b>always</b> make it <b>external</b>. Otherwise Hive will delete the HDFS files when you drop your table, which 99.99999% of the time is not what you want to do.</div>

# %%
# Drop the epd expression table
query = """
    drop table if exists {0}.epd_expr_csv
""".format(
    username
)
cur.execute(query)

# Create the epd expression table
query = """
    create external table {0}.epd_expr_csv(
        promoter_name string,
        expression int,
        position_vs_TSS int,
        sample string
    )
    row format delimited fields terminated by ','
    stored as textfile
    location '/data/epd/csv/promoter_expression'
""".format(
    username
)
cur.execute(query)

# %% [markdown]
# Now verify that your table was properly created.
#
# **Do not** remove the limit, or use a very large one!!
#

# %%
query = """
    select * from {0}.epd_expr_csv limit 5
""".format(username)
cur.execute(query)
for result in cur.fetchall():
    print(result)

# %% [markdown]
# **Notes**: When displaying `select` results, we could also use `pd.read_sql`, which requires two arguments, the query `query` and the connection `conn` (not the cursor `cur`) and returns a dataframe for review. You can check [here](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.read_sql.html?highlight=read_sql) for additional details about this function.

# %%
pd.read_sql(query, conn)

# %% [markdown]
# **Q2**: Note the first row above. What did we do wrong? Can you add a table property to the table creation query in order to solve the problem? (hint: use tblproperties)
#
# **Solutions:**
# * You must drop the headers - use the little documented `tbleproperties("skip.header.line.count="1")`
# * See other common properties available in [tbleproperties](https://cwiki.apache.org/confluence/display/hive/LanguageManual+DDL#LanguageManualDDL-CreateTable). Also different `SerDe` (Serializer, Deserializer) such as OpenCSV, ORC SerDe[[1](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+ORC),[2](https://orc.apache.org/docs/hive-config.html)], Json SerDe, and Parquet Hive SerDe, will have their own properties.

# %%
# Drop the epd sequences table
query = """
    drop table if exists {0}.epd_expr_csv
""".format(
    username
)
cur.execute(query)

# Create the epd sequences table
query = """
    create external table {0}.epd_expr_csv(
        promoter_name string,
        expression int,
        position_vs_TSS int,
        sample string
    )
    row format delimited fields terminated by ','
    stored as textfile
    location '/data/epd/csv/promoter_expression'
    tblproperties ("skip.header.line.count"="1")
""".format(
    username
)
cur.execute(query)
query = """
    select * from {0}.epd_expr_csv limit 5
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# **Q3**: Load the table `/data/epd/csv/promoter_samples_expression` as `epd_sam_csv`, and use it to select promoters in `epd_expr_csv` which are expressed in at least 100 samples.
#

# %%
# Drop the epd samples expression table
query = """
    drop table if exists {0}.epd_sam_csv
""".format(
    username
)
cur.execute(query)

# Create the epd samples expression table
query = """
    create external table {0}.epd_sam_csv(
        promoter_name string,
        num_active_samples string,
        average_expression float
    )
    row format delimited fields terminated by ','
    stored as textfile
    location '/data/epd/csv/promoter_samples_expression'
    tblproperties ("skip.header.line.count"="1")
""".format(
    username
)
cur.execute(query)
query = """
    select * from {0}.epd_sam_csv limit 5
""".format(username)
pd.read_sql(query, conn)


# %%
query = """
select promoter_name, expression, sample
from {0}.epd_expr_csv as expr
join (
        select promoter_name as id, num_active_samples, average_expression
        from {0}.epd_sam_csv as sam
        where num_active_samples > 100
    ) sa
on expr.promoter_name = sa.id
where sample like "%cyte %"

""".format(username)
df = pd.read_sql(query, conn)

# %% [markdown]
# **Q4**: Use PCA on the table extracted with hive to visualize samples in 2 dimensions based on their expression values. Are there samples which stand apart from the others ?
#
# > Now that we have extracted a smaller subset from the data, we can use pandas and sklearn for this exploratory analysis

# %%
samples = df.pivot_table(
    values='expression',
    index='sample',
    columns='promoter_name',
    fill_value=0
)
cell_type = samples.index.str.split('-').str[0]

# %%
from sklearn.decomposition import PCA
pca = PCA(whiten=True).fit(samples.to_numpy())
pcs = pca.transform(samples.to_numpy())

# %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 9))
sns.scatterplot(pcs[:, 0], pcs[:, 1])
for i in range(pcs.shape[0]):
    if pcs[i, 0] > 1:
        plt.text(pcs[i, 0], pcs[i, 1], cell_type[i][:10])
plt.show()

# %%
# Getting top 5 promoters with highest contribution to PC1
samples.columns[np.argsort(pca.components_[0])][-5:]

# %%
# Alternative query without auxiliary table
query = """
with high_expr as 
(
    select promoter_name as good_name, 
               count(*) as num_samples
    from {0}.epd_expr_csv
    group by promoter_name
    having num_samples > 100
    )
select promoter_name, expression, sample
from {0}.epd_expr_csv expr
join high_expr
on expr.promoter_name = high_expr.good_name
limit 5
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
#
# # Let's try different formats now
#
# The same data is stored under three other formats, CSV, Bz2 CSV, ORC and PARQUET.
#
# You have already created a table using CSV. Can you create the tables epd_expr_bz2, epd_expr_orc, and epd_expr_parquet for the other three formats respectively?
#
# For each format, time how long it takes to count the number of rows with the magic function `%%time` (You can also use `%timeit` to get a more accurate estimate, but it will take longer. It runs the instructions many times and prints the mean and standard deviation or durations).
#
# * Which storage format has the best read performances?
# * Compare the size versus performance tradeoffs using their storage format respective footprints (sizes in bytes) calculated in the previous exercise?
#
#
# **Solutions:**
#
# * First you must find the location of the external tables. The hint is in the earlier exercises which points to the location of `/data/epd/csv/`. A bit of `hdfs dfs -ls /data/epd` exploration show that that the other tables are respectively under the HDFS directories `/data/epd/csv/`, `/data/epd/orc` and `/data/epd/parquet` respectively.
# * The **CSV** format is easy. The `create table` similar to the **Bz2** format shown earlier, but for the location which is `/data/epd/csv`.
# * For **ORC** and **PARQUET** you need to understand the storage format specified by the `stored as` directive[[3](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-StorageFormatsStorageFormatsRowFormat,StorageFormat,andSerDe)].
#
# **Results:**
# * To save you time we have included under each query, the output of the query executed using the `beeline` command line interface in a terminal. The command provides additional details showing the time spent in each task during the execution of the query.
# * The running times will vary - they depend on the load at the time the command is being executed, and the ability of the application master to negotiate worker containers that are closest to the data.
# * All tables return __3,860,230__ rows.
# * Note that the total CPU time is (much) lower the Wall time. This is because most of the work happens outside the docker container of this jupyter notebook.
# * We observe that **ORC** and **PARQUET** are faster than **CSV** or **Bz2**. **ORC** tends to be slightly faster than **PARQUET** for large datasets, and **CSV** is significantly faster than **Bz2** (2 seconds versus 8 seconds average). Remember that the **Bz2** compression saves storage space and network IO as seen in the previous exercise, however decompressing the data must be paid with CPU, and there is thus a tradeoff. In this case the uncompressed **CSV** format wins over **Bz2**.
# * **Hint:** you can get more info about the table using the Hive query `describe formatted db_name.table_name}`[[4](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-Describe)]

# %%
expr_drop_query = """
    drop table if exists {0}.epd_expr_{1}
"""

# %%
cur.execute(expr_drop_query.format(username, 'csv'))

query = """
    create external table {0}.epd_expr_csv(
        promoter_name string,
        expression int,
        position_vs_TSS int,
        sample string
    )
    row format delimited fields terminated by ','
    stored as textfile
    location '/data/epd/csv/promoter_expression'
    tblproperties ("skip.header.line.count"="1")
""".format(username)
cur.execute(query)

# %%
# %%time
query = """
    select count(*) from {0}.epd_expr_csv
""".format(username)
pd.read_sql(query, conn)

# %% [raw]
# ----------------------------------------------------------------------------------------------
#         VERTICES      MODE        STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED  
# ----------------------------------------------------------------------------------------------
# Map 1 .......... container     SUCCEEDED      1          1        0        0       0       0  
# Reducer 2 ...... container     SUCCEEDED      1          1        0        0       0       0  
# ----------------------------------------------------------------------------------------------
# VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 5.19 s     
# ----------------------------------------------------------------------------------------------
# INFO  : Status: DAG finished successfully in 5.02 seconds
# INFO  : 
# INFO  : Query Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : OPERATION                            DURATION
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : Compile Query                           0.25s
# INFO  : Prepare Plan                            0.23s
# INFO  : Get Query Coordinator (AM)              0.00s
# INFO  : Submit Plan                             0.17s
# INFO  : Start DAG                               0.54s
# INFO  : Run DAG                                 5.02s
# INFO  : ----------------------------------------------------------------------------------------------

# %%
cur.execute(expr_drop_query.format(username, 'bz2'))

query = """
    create external table {0}.epd_expr_bz2(
        promoter_name string,
        expression int,
        position_vs_TSS int,
        sample string
    )
    row format delimited fields terminated by ','
    stored as textfile
    location '/data/epd/bz2/promoter_expression'
    tblproperties ("skip.header.line.count"="1")
""".format(username)
cur.execute(query)

# %%
# %%timeit
query = """
    select count(*) from {0}.epd_expr_bz2
""".format(username)
pd.read_sql(query, conn)

# %% [raw]
# ----------------------------------------------------------------------------------------------
#         VERTICES      MODE        STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED  
# ----------------------------------------------------------------------------------------------
# Map 1 .......... container     SUCCEEDED      1          1        0        0       0       0  
# Reducer 2 ...... container     SUCCEEDED      1          1        0        0       0       0  
# ----------------------------------------------------------------------------------------------
# VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 10.37 s    
# ----------------------------------------------------------------------------------------------
# INFO  : Status: DAG finished successfully in 10.15 seconds
# INFO  : 
# INFO  : Query Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : OPERATION                            DURATION
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : Compile Query                           0.19s
# INFO  : Prepare Plan                            0.29s
# INFO  : Get Query Coordinator (AM)              0.00s
# INFO  : Submit Plan                             0.19s
# INFO  : Start DAG                               0.56s
# INFO  : Run DAG                                10.15s
# INFO  : ----------------------------------------------------------------------------------------------

# %%
cur.execute(expr_drop_query.format(username, 'orc'))

query = """
    create external table {0}.epd_expr_orc(
        promoter_name string,
        expression int,
        position_vs_TSS int,
        sample string
    )
    stored as orc
    location '/data/epd/orc/promoter_expression'
""".format(username)
cur.execute(query)

# %%
# %%time
query = """
    select count(*) from {0}.epd_expr_orc
""".format(username)
pd.read_sql(query, conn)

# %% [raw]
# ----------------------------------------------------------------------------------------------
#         VERTICES      MODE        STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED  
# ----------------------------------------------------------------------------------------------
# Map 1 .......... container     SUCCEEDED      1          1        0        0       0       0  
# Reducer 2 ...... container     SUCCEEDED      1          1        0        0       0       0  
# ----------------------------------------------------------------------------------------------
# VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 4.48 s     
# ----------------------------------------------------------------------------------------------
# INFO  : Status: DAG finished successfully in 4.24 seconds
# INFO  : 
# INFO  : Query Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : OPERATION                            DURATION
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : Compile Query                           0.27s
# INFO  : Prepare Plan                            0.29s
# INFO  : Get Query Coordinator (AM)              0.00s
# INFO  : Submit Plan                             0.20s
# INFO  : Start DAG                               0.54s
# INFO  : Run DAG                                 4.24s
# INFO  : ----------------------------------------------------------------------------------------------
#

# %%
cur.execute(expr_drop_query.format(username, 'parquet'))

query = """
    create external table {0}.epd_expr_parquet(
        promoter_name string,
        expression int,
        position_vs_TSS int,
        sample string
    )
    stored as parquet
    location '/data/epd/parquet/promoter_expression'
""".format(username)
cur.execute(query)

# %%
# %%time
query = """
    select count(*) from {0}.epd_expr_parquet
""".format(username)
pd.read_sql(query, conn)

# %% [raw]
# ----------------------------------------------------------------------------------------------
#         VERTICES      MODE        STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED  
# ----------------------------------------------------------------------------------------------
# Map 1 .......... container     SUCCEEDED      1          1        0        0       0       0  
# Reducer 2 ...... container     SUCCEEDED      1          1        0        0       0       0  
# ----------------------------------------------------------------------------------------------
# VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 4.84 s     
# ----------------------------------------------------------------------------------------------
# INFO  : Status: DAG finished successfully in 4.62 seconds
# INFO  : 
# INFO  : Query Execution Summary
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : OPERATION                            DURATION
# INFO  : ----------------------------------------------------------------------------------------------
# INFO  : Compile Query                           0.19s
# INFO  : Prepare Plan                            0.28s
# INFO  : Get Query Coordinator (AM)              0.00s
# INFO  : Submit Plan                             0.20s
# INFO  : Start DAG                               0.57s
# INFO  : Run DAG                                 4.62s
# INFO  : ----------------------------------------------------------------------------------------------
#
