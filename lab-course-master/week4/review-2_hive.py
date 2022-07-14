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
# # Data Exploration with Hive
#
# In the next set of exercises, you will use Hive to perform an exploratory analysis of data published by the [Swiss Public Transport Open Data Platform](https://opentransportdata.swiss) based on SBB/CFF/FFS data.
#
# This is the second part of a two-parts revision notebooks about Hive. The notebooks are designed so that they can be reviewed as standalone materials.

# %% [markdown]
# ---
# ### About the data
#
# You will have two data sets:
#  
# **1. Istdaten** (real data) Data source: <https://opentransportdata.swiss/en/dataset/istdaten>.
#
# *Description:* This data contains the actual arrival and departure times of trains, busses and other types of public transports observed at many locations across switzerland (and bordering countries) from 2018 to end of 2021.
#
# *Data location:* The data can be found on HDFS under `/data/sbb/orc/istdaten`.
#
# *Format:* The dataset is originally presented as a collection of textfiles with fields separated by `;` (semicolon). For efficiency, the textfiles have been compressed into Optimized Row Columnar ([ORC](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+ORC)) file format using the data import and transform methods described in the [review-1_hive.py](./review-1_hive.py) notebook.
#
# The full description of the data can be found in <https://opentransportdata.swiss/de/cookbook/ist-daten/> in four languages.
#
# Each line of the table represents a stop and contains the actual arrival and departure times of a public transport journey that has completed. The line also contains the expected arrival and departure times of the journey from the time tables.
#
# Arrival and departure delays are calculated using the difference between the actual times and the time table. Actual times are measured in different manner depending on the stop location. It is either the effective actual data or last forecast received by the customer information system, which is less accurate. If no real-time information was available, the relevant journey is not included in the table.
#
# The row has other information related to the journey in question, here are the relevant column descriptions:
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
# When the stop is the start or end of a journey, the corresponding columns will be empty (`ANKUNFTSZEIT`/`ABFAHRTSZEIT`).
#
# The manner in which the time is measured is indicated by the `AN_PROGNOSE_STATUS` and `AB_PROGNOSE_STATUS` fields. Possible values are:
# - `UNBEKANNT` - Unknown, no forecast or actual times available for this and all previous stops
# - `''` - Empty, same as `PROGNOSE`
# - `PROGNOSE` - Time forecasted from the SBB operational planning's system.
# - `GESCHAETZT` - Calculated actual arrival time.
# - `REAL` - Effective actual arrival time
#
# In some cases, the actual times were not measured so the `AN_PROGNOSE_STATUS`/`AB_PROGNOSE_STATUS` will be empty or set to `PROGNOSE` and `AN_PROGNOSE`/`AB_PROGNOSE` will be empty.
#
#
# **2. Geostops**
#
# *Description:* The geostops data contains the information about the stops geographical locations.
#
# *Data location:* The data is available in HDFS under `/data/sbb/orc/allstops`.
#
# *Format:* Here are the relevant column descriptions:
#
# - `STOP_ID`: numerical ID of the stop
# - `STOP_NAME`: actual name of the stop
# - `STOP_LAT`: the latitude (in WGS84 coordinates system) of the stop
# - `STOP_LON`: the longitude of the stop
# - `LOCATION_TYPE`: set to 1 (True) if the stop is a cluster of stops, such as a train station or a bus terminal.
# - `PARENT_STATION`: If the stop is part of a cluster, this is the `STOP_ID` of the cluster.
#
# A majority of `HALTESTELLEN_NAME` in the **Istdaten** data set have a matching `STOP_NAME` in **geostop**. However, this is often not always the case: the names are sometime spelled differently (e.g. abbreviations), or expressed in different languages. In many cases, the `STOP_ID` is even used to identify the stop in `HALTESTELLEN`. To complicate matters, character encodings were used throughout the years, and a stop name does not always compare the same under different encodings.

# %% [markdown]
# ---
# ## About Apache Hive, and the Hive Query Language
#
# Apache Hive is a data warehouse software project built on top of Apache Hadoop for providing data query and analysis. Hive gives an SQL-like interface to query data stored in various databases and file systems that integrate with Hadoop. 
#
# The storage and querying operations of Hive closely resemble those of traditional databases. However, while based on SQL, HiveQL does not strictly follow the full SQL standard. The differences are mainly because Hive has to comply with the restrictions of Hadoop and MapReduce. 
#
# The Hive query language provides an SQL-like [SELECT](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Select) statement for querying the tables.
#
# Before we go further we illustrate the queries you are about to run in this exercise. For this we use a simplified view of a fictitious SBB time table, which we call **sbb_timetable**:
#
# |time|mode|service|city|delay|
# |-|-|-|-|-|
# |00:00|train|IC723|lausanne|25|
# |00:05|train|S6|zurich|5|
# |00:10|Bus|77|zurich|10|
# |00:20|train|IC723|geneva|30|
# |05:00|bus|66|lausanne|5|
# |05:05|train|IC719|geneva|10|
#
# The table has 5 _columns_ and 6 _rows_.
#
# Using the Hive select command you can:
#
# **1. Select a subset of columns, or create new columns**
#
# E.g.: `SELECT mode, city, delay FROM sbb_timetable`
#
# |mode|city|delay|
# |-|-|-|
# |train|lausanne|25|
# |train|zurich|5|
# |Bus|zurich|10|
# |train|geneva|30|
# |bus|lausanne|5|
# |train|geneva|10|
#
# * You can use aliases for the columns using the `AS column_name` clause.
# * The `SELECT` statement may include Operators and User Defined Functions [(UDF)](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF) on the columns such as `lower(mode)`.
# * You can also calculate new columns, e.g. `SELECT mode, city, round(delay/60) AS delay_minute FROM sbb_timetable`.
#
# **2. Select only rows that match a condition**
#
# E.g.: `SELECT * FROM sbb_timetable WHERE lower(mode)='bus'`
#
#
# |time|mode|service|city|delay|
# |-|-|-|-|-|
# |00:10|Bus|77|zurich|10|
# |05:00|bus|66|lausanne|5|
#
# * The `*` selects all the columns of the FROM table.
# * The part after the `WHERE` statement may include Operators and User Defined Functions [(UDF)](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF), such as `lower(mode)='bus' AND round(delay/10) > 1`.
#
# **3. Group the rows and replace their values with aggregated values**
#
# E.g. `SELECT lower(mode), count(*) AS num, avg(delay) AS avg_delay FROM sbb_timetable GROUP BY lower(mode)`
#
# |mode|num|avg_delay|
# |-|-|-|
# |train|4|17.5|
# |bus|2|3.75|
#
# Group the rows based on the values of column names specified in the `GROUP BY` statements (`mode` in this example). Input rows that have the same value are grouped together, such that the result of the query results in one row for each group. Typically, grouping is used to apply some sort of aggregate function for each group. In this example it computes the average delay of each group.
#
# * The part after the `GROUP BY` statement may include User Defined Functions [(UDF)](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF) applied to any number of table columns, such as `GROUP BY lower(mode)='bus',round(delay/10)`.
# * Columns returned by the `SELECT` statement that are not part of the `GROUP BY` statement must be aggregated using an aggregate function [(UADF)](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-Built-inAggregateFunctions(UDAF)), so that they have one result per group.
# * An UADF aggregate functions in a `SELECT` without a `GROUP BY` applies to all the rows in the table. You would use it for instance to compute the average of a column or count the number of rows.
# * Use `WHERE` to select rows before grouping them. Use `HAVING` to select groups that satisfy an aggregate condition. E.g.: `SELECT city,avg(delay) FROM sbb_timetable WHERE delay>0 GROUP BY city HAVING avg(delay)>60`.
#
# **4. Combines rows of two or more tables**
#
# Assume you have another table **sbb_city** containing the GPS coordinates of the city centers:
#
# |Name|Latitude|Longitude|
# |-|-|-|
# |lausanne|46.5198|6.6335|
# |geneva|46.2|6.15|
# |zurich|47.366667|8.55|
#
# A [JOIN](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Joins) clause can be used to combine rows of the two (or more) tables, based on related column between them.
#
# E.g.: `SELECT time, mode, service, city, delay, latitude, longitude FROM sbb_timetable JOIN sbb_city ON sbb_timetable.city = sbb_city.name`
#
# |time|mode|service|city|delay|latitude|longitude|
# |-|-|-|-|-|-|-|
# |00:00|train|IC723|lausanne|25|46.5198|6.6335|
# |00:05|train|S6|zurich|5|47.366667|8.55|
# |00:10|Bus|77|zurich|10|47.366667|8.55|
# |00:20|train|IC723|geneva|30|46.2|6.15|
# |05:00|bus|66|lausanne|5|46.5198|6.6335|
# |05:05|train|IC719|geneva|10|46.2|6.15|
#
#
# 📝 The SQL language is case-insensitive. In the following we will apply the generally accepted SQL convention, and use upper case for SQL keywords, and lower case for user-defined table and column names.

# %% [markdown]
# ---
# #### 1. Preamble
#
# Most of the step below, you have already seen in the previous notebook [review-1_hive.py](./review-1_hive.py). We will only enumerate the steps here:
#
# * _Notebook initialization_: set the important global variables.
# * _Connect to Hive_: create a connection object that you can use to interact with the remote Hive.
# * _Create your database_: declare a personal space under which you will store your table definitions.
#
# ##### 1.1 Notebook initialization

# %%
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

hiveaddr = os.environ['HIVE_SERVER2']
username = os.environ['RENKU_USERNAME']
print(hiveaddr)
print(username)

# %% [markdown]
# ##### 1.2. Connect to Hive

# %%
from pyhive import hive

(hivehost,hiveport)=hiveaddr.split(':')
conn = hive.connect(host=hivehost,port=hiveport,username=username)
cursor = conn.cursor()

# %% [markdown]
# ##### 1.3. Create your database, and make it your defaults

# %%
query = "CREATE DATABASE IF NOT EXISTS {0}".format(username)
cursor.execute(query)

# %%
query = "USE {0}".format(username)
cursor.execute(query)

# %% [markdown]
# Create an **external** SBB table of real time data (istdaten). Drop the table first if it exists.

# %%
query = """
    DROP TABLE IF EXISTS {0}.sbb_orc_istdaten
""".format(username)
cursor.execute(query)

# %%
query = """
    CREATE EXTERNAL TABLE {0}.sbb_orc_istdaten(
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
    LOCATION '/data/sbb/orc/istdaten/'
""".format(username)
cursor.execute(query)

# %% [markdown]
# ##### 1.4. About the SBB Table
#
# In the following exercises you will be using an external Hive table of SBB data. The table was created using exactly the same method as previously seen in the [revew-1_hive.py](./review-1_hive.py) notebook. The main difference is the size of the table - spanning the years 2018 to 2021, it contains several Billions of rows.

# %% [markdown]
# Show all the tables defined in your database, and verify that the `sbb_orc_istdaten` table is there:

# %%
cursor.execute('SHOW TABLES in {0}'.format(username))
cursor.fetchall()

# %% [markdown]
# Describe the schema of the table

# %%
cursor.execute('DESCRIBE EXTENDED {0}.sbb_orc_istdaten'.format(username))
cursor.fetchall()

# %% [markdown]
# ---
# #### 2. Query the table
#
# Print a few lines to get an idea of table's content.
#
# ⚠️ Do not always assume that a few lines of the table are representative of its full content - this is big data, spanning several years, and many things can change over that period of time. We have seen cases for instance where the date field would vary over time from _YYYY/mm/dd_, to _mm/dd/YYYY_ and to _dd/mm/YYYY_. 

# %%
import pandas as pd
pd.read_sql('SELECT * FROM {0}.sbb_orc_istdaten LIMIT 5'.format(username), conn)

# %% [markdown]
# Count the total number of rows in the table (see [COUNT](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-Built-inAggregateFunctions(UDAF))).
#
# Note: `%%time` is a Jupyter notebook magic. It is used to measure and display the time spent running the code in the cell.

# %%
# %%time
pd.read_sql('SELECT count(*) FROM {0}.sbb_orc_istdaten'.format(username), conn)

# %% [markdown]
# Group by `produkt_id` (modes of transport) and count the number of rows for each mode

# %%
# %%time
pd.read_sql(
        "SELECT produkt_id, count(*) FROM {0}.sbb_orc_istdaten GROUP BY produkt_id".format(username),
        conn)

# %% [markdown]
# There are only 10 distinct values possible of `produkt_id` in the whole table. Note however that a significant number of produkt identifiers are invalid (`None` or empty `''`). Also, bus transportation is identified multiple times due to an inconsistent use of upper and lower case letters.
#
# We thus modify the query and use `produkt_id IS NOT NULL AND produkt_id <> ''` to drop the invalid values. We also convert all the `produkt_id` values to lower case with `lower(produkt_id)` to correct for inconsistent use of upper and lower case letters. We also give a proper name to the count column with `AS num` instead of the automtatically generated name `_c1`.
#
# The more curious will also try to use the `SELECT * FROM sbb_orc_istdaten WHERE produkt_id = '244'` command to better understand where this unique value of `244` comes from, and whether it should be ignored.

# %%
# %%time
query = """
SELECT lower(produkt_id) AS mode, count(*) AS num
    FROM {0}.sbb_orc_istdaten 
    WHERE
        produkt_id IS NOT NULL
        AND produkt_id <> ''
    GROUP BY lower(produkt_id)
""".format(username)
pd.read_sql(query,conn)

# %% [markdown]
# In the same vein, we can count how many distinct public transport stops (`stop_name`) this table contains, and how many times (`num`) each one appears. We display the list in decreasing order of occurence with `ORDER BY num DESC`. 
#
# ⚠️ Hive supports also the [SORT BY](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+SortBy) syntax, which offers weaker ordering guarantees than `ORDER BY` but is faster (see [Difference between SORT BY and ORDER BY](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+SortBy#LanguageManualSortBy-DifferencebetweenSortByandOrderBy))

# %%
# %%time
query = """
SELECT lower(haltestellen_name) AS stop_name, count(*) AS num
    FROM {0}.sbb_orc_istdaten
    GROUP BY lower(haltestellen_name)
    ORDER BY num DESC
""".format(username)
pd.read_sql(query,conn,index_col='stop_name')

# %% [markdown]
# ---
# #### 3. Compute public transport delays in Switzerland

# %% [markdown]
# We will now demonstrate some of advanced features of HiveQL in a practical use case. You are about to visualize public transport delay statistics in the course of a typical weekday on a map of Switzerland. 
#
# Most of the information you need for this visualization can be generated as a pipeline of SELECT operations in one Create-Table-As-Select ([CTAS](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTableAsSelect(CTAS))) statement.
#
# However, for the sake of clarity, we break down this one big CTAS statement into 3 separate statements.
#
# To achieve this we will be using a `VIEW`s instead of `TABLE`s. Whereas tables contain data, views are just `SELECT` statements that get combined and executed as a single one. Think of them as _functions_ in traditional programming language. Tables and views are created in a similar way (just substitute `TABLE` for `VIEW` in the next queries). 
# The advantage of using tables is that intermediate results are saved, and you do not need to execute all the select statements in order to arrive to a final result. The advantage of views, is that you have the guarantee that the date is up to date, and is not coming from an outdated intermediate table. They also do not use storage space for intermediate results.
#
# ---
# **VIEW Statement 1** - extract delays information from Istdaten
#
# You want all the delays at every stop for every category of transport measured over the considered time period. All the information you need is found in the **SBB Istdaten** table, you only need to compute the difference between the actual times of arrival and the expected arrival times from the time tables (we could also use the departure time).
#
# Use table `sbb_orc_istdaten` to create a new table `sbb_times` containing the following 4 columns:
# * `stop_name`: the name of a stop.
# * `expected`: a big integer indicating the expected time of arrival (from the SBB time table) of a public transport, in seconds since _1970.01.01 00:00_ at the given `stop_name`.
# * `actual`: a big integer indicating the time at which the transport actually arrived.
# * `mode`: the mode of that transport (bus, train, etc.), in lower case.
#
# The output should look something like this:
#
# |mode|stop_name|expected|actual|
# |-|-|-|-|
# |zug|Caslano|1558155420|1558155444|
# |zug|Magliaso|1558155540|1558155550|
# |zug|Magliaso Paese|1558155600|1558155658|
# |zug|Agno|1558155840|1558155792|
#
# As mentioned earlier, you will create views using the - `CREATE VIEW viewname ... AS SELECT ...` statement.
#
# You should understand most of the query below - `mode` is the lower case value of the `produkt_id` from the `sbb_orc_istdaten` table, and `stop_name` is a new name for `haltestellen_name`. In the `WHERE` statement we ignore rows with null or empty `produkt_id`, `haltestellen_nam`, and `ankuftszeit` fields.
#
# The `an_prognose_status in ('REAL', 'GESCHAETZT')` is self describing, it only selects rows whose `an_prognose_status` is either `REAL` (real, as observed) or `GESCHAETZT` (estimated indirectly from other real-time observations).
#
# The function `unix_timestamp(colname, 'dateformat')` is new to you. It is a [Date UDF](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-DateFunctions) function, and it is used to convert a date given in a string format (in this case _Year.month.day hour:minutes_) into a large integer corresponding to the number of seconds elapsed from _1970.01.01 00:00_ to the date. In this example, we convert the fields `ankunftszeit` and `an_prognose` of the table `sbb_orc_istdaten`. We call them `expected` and `actual` respectively in the new table. Normally, you should be vigilant about the time zone information, and pay attention to the differences between summer and winter time. However, for simplicity, we assume that all timestamps are from a unique time zone.
#
# Also new to you is the `RLIKE` statement. It works by selecting rows whose fields match a given pattern. Without going into detail, in this example, suffice it to say that we are only considering the spring months of March through June 2019.

# %%
# %%time
cursor.execute("DROP VIEW IF EXISTS {0}.sbb_times".format(username))
query = """
CREATE VIEW {0}.sbb_times
    AS SELECT
        lower(produkt_id) AS mode,
        haltestellen_name AS stop_name,
        unix_timestamp(ankunftszeit, 'dd.MM.yyyy HH:mm') AS expected, 
        unix_timestamp(an_prognose,  'dd.MM.yyyy HH:mm:ss') AS actual
    FROM {0}.sbb_orc_istdaten      
    WHERE
        produkt_id <> ''
        AND produkt_id IS NOT NULL
        AND ankunftszeit IS NOT NULL
        AND an_prognose_status in ('REAL', 'GESCHAETZT')
        AND betriebstag RLIKE '^..\.0[3456].2019'
        AND haltestellen_name <> ''
        AND haltestellen_name IS NOT NULL
""".format(username)
cursor.execute(query)

# %% [markdown]
# You can verify the output of this statement:

# %%
# %%time
import pandas as pd
pd.read_sql("SELECT * FROM {0}.sbb_times limit 5".format(username),conn)

# %% [markdown]
# ---
# **VIEW Statement 2** - compute the delay interquartiles
#
# You want to use the historical data selected in VIEW Statement 1 to estimate a meaure of the expected delay at a given stop for a given category of transport at different moments of the day. This is the same as using your commute experience to build a mental picture of delay expectations, under various conditions, but using rigorous statistical methods. In this example we first start with a train station at a given time of the day, and we ask ourselves what were the maximum of the 10, 25, 50, 75 and 90 shortest delays out of every 100 trains (from any direction) that stopped there in the recent past. We call these numbers the 10th, 25th, 50th (or median), 75th and 90th delay percentiles. We then repeat the question for every other mode of transports at that location (bus, trams, etc), for every other time of day (in 15 minutes intervals), and finally for every stop location. This information can be valuable the next time you plan your commute. Assuming that the past repeats itself, you expect the odds to be 1 out of 4 that your train will not be delayed by more than the 25th percentile, 1 out of 2 that it will delayed by less than the 50th percentile and so on.
#
# Use table `sbb_times` to create a new table `sbb_orc_delays` containing the following 4 columns:
#
# * `mode`: mode of transport.
# * `stop_name`: name of a public transport stop.
# * `time_of_day`: a number between 0 and 95, representing a daily slice of 15 minutes.
# * `delay`: an array containing respectively the 10th, 25th, 50th (median), 75th and 90th delay percentiles (in seconds) of the given mode of transport at that stop during the specified 15 minutes interval.
#
# The output should look something like this:
#
# |mode|stop_name|time_of_day|delay|
# |-|-|-|-|
# |zug|Caslano|26|\[6.60,16.5,33.0,68.0,89.0\]|
# |zug|Magliaso|27|\[41.3,50.75,66.5,82.25,91.7\]|
#
#
# Note: you will divide the day into 15min intervals and calculate the delay percentiles calculated in each interval over a period of 4 months  (business days only), for each mode of transport and each stop. Breaking the day into 15mins intervals allows us to capture delay variations at different moments of the day (such as rush hours).
#
# You want to compute the delays statistics for every distinct slice of 15 minute interval, stop name, and mode of transport. This is a textbook example for the `GROUP BY` query.
#
# Additional details about the query:
#
# 1. `floor(expected/900)%96` converts the number of seconds since the _1970.01.01 00:00_ into daily 15min (900 seconds) intervals. The `%96` part means _modulo 96_, the number of 15 minute intervals in a day. Therefore _time_of_day=0_ corresponds to the first 15 minutes of the day, 1 is the next 15mins, 95 the last 15 minutes, then the first 15 minutes of the next day is 0 again and so on.
# 2. `if(actual > expected, actual - expected, 0)` computes the delay between the expected time of arrival and the actual time of arrival, and 0 if the public transport left eariler than expected (we do not count negative delays).
# 3. `percentile(delays,p)` is one of the Hive built-in aggregate functions ([UDFA](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-Built-inAggregateFunctions(UDAF))), used to aggregate values in a group. In this example it computes the p-percentiles of the delays calculated above in each group, i.e. for each distinct combination of mode of transport, stop, and time of day (in 15 minutes intervals).
# 4. You want the $10^{th}$, $25^{th}$, $50^{th}$ (median), $75^{th}$ and $90^{th}$ percentiles. This is done by specifying the list of percentiles using the `ARRAY(0.1,0.25,0.5,0.75,0.9)` notation for the `p` argument of the `percentile` aggregate function above. Combining 2, 3, and 4 we get `percentile(IF(actual > expected, actual - expected, 0),ARRAY(0.1,0.25,0.5,0.75,0.9))`.
# 5. You want the delays to be for working days only, or to put it another way, you want to remove the weekend data. To do this you must know which day of the week the `expected` seconds after 1970.01.01 happens. The function `extract(dayofweek from time)` will help here. You can try it yourself and see how it works with the query statement `SELECT extract(dayofweek from "2021-04-16")`. The function returns a value between 0 and 6 corresponding respectively to Saturday to Friday. We only consider values `NOT IN (0,1)`, or put it simply, we ignore Saturdays and Sundays.
# 6. The time argument of the `extract` function must be of type `TIMESTAMP`, or a `STRING` representing a time in a format like `YYYY-mm-dd hh:mm:ss.sss`. Since the `expected` value is an integer, we cast it into a `TIMESTAMP`, using the `CAST(expected*1000 AS TIMESTAMP)` function. Note the twist, we must convert `expected` into milliseconds since _1970.01.01 00:00_.

# %%
# %%time
cursor.execute("DROP VIEW IF EXISTS {0}.sbb_orc_delays".format(username))
query = """
    CREATE VIEW {0}.sbb_orc_delays
    AS SELECT
        mode,
        stop_name,
        floor(expected/900)%96 AS time_of_day,
        percentile(IF(actual > expected, actual - expected, 0),ARRAY(0.1,0.25,0.5,0.75,0.9)) AS delay_percentile
    FROM {0}.sbb_times
    WHERE
        extract(dayofweek from CAST(expected*1000 AS TIMESTAMP)) NOT IN (0,1)
        GROUP BY mode,stop_name,floor(expected/900)%96
""".format(username)
cursor.execute(query)

# %% [markdown]
# You can verify the output of the aggregation:

# %%
# %%time
pd.read_sql("SELECT * FROM {0}.sbb_orc_delays limit 5".format(username),conn)

# %% [markdown]
# ---
# **VIEW Statement 3**
#
# We have computed the percentiles, we would like now to place them on a map of Switzerland, so that we have a better understanding of when but also where (geographically) they happen.
#
# So far none of the data we have used includes the geographic location of the stops, which we need for the task.
#
# We must add this information ourselves from another source. In this case we create the the `sbb_orc_stops` table, available from HDFS on `/data/sbb/orc/allstops`.

# %%
cursor.execute("DROP TABLE IF EXISTS {0}.sbb_orc_stops".format(username))
query = """
create external table {0}.sbb_orc_stops(
        STOP_ID        string,
        STOP_NAME      string,
        STOP_LAT       double,
        STOP_LON       double,
        LOCATION_TYPE  string,
        PARENT_STATION string
    )
    stored as orc
    location '/data/sbb/orc/allstops'
    tblproperties ('orc.compress'='SNAPPY','immutable'='true')
""".format(username)
cursor.execute(query)

# %% [markdown]
# You can verify the schema of`sbb_orc_stops` with a `DESCRIBE` command

# %%
cursor.execute("DESCRIBE {0}.sbb_orc_stops".format(username))
cursor.fetchall()

# %% [markdown]
# And display a preview of its content

# %%
pd.read_sql("SELECT * FROM {0}.sbb_orc_stops limit 5".format(username),conn)

# %% [markdown]
# On one hand we have delay percentiles in a table `sbb_orc_delays` and on the other hand you have the latitudes and longitudes of the stops where the delays take place in table `sbb_orc_stops`.
#
# Both tables use one or more fields to identify the stops. They are `sbb_orc_delays.stop_name`, `sbb_orc_stops.stop_name` and `sbb_orc_stops.stop_id`. By matching the stop names it is possible to associate the delays of the first table with their geolocations in the second. But there is a twist: the transport operators who provide the `sbb_orc_delays.stop_name` inputs are not always consistent and they sometime use the numerical identifier of the stop and sometime its name. Therefore a `sbb_orc_delays.stop_name` could either match `sbb_orc_stops.stop_name` or `sbb_orc_stops.stop_id`. To complicate matters, the tables may refer to a same stop under different names.
#
# Nethertheless, this is sufficient for the intended purpose. You must just accept that a few stops will be left-out if you cannot find a match.
#
# This is done using the Hive QL [SELECT ... JOIN ... ON](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Joins) operation. There are many variations of the `JOIN` statement (`FULL`, `LEFT`, `RIGHT`,  `OUTER`, ...), but we will not go into details here.

# %%
# %%time
cursor.execute("DROP VIEW IF EXISTS {0}.sbb_orc_delays_join".format(username))
query = """
    CREATE VIEW {0}.sbb_orc_delays_join
    AS SELECT DISTINCT
        a.mode,
        a.stop_name,
        a.time_of_day,
        a.delay_percentile[0] AS p10,
        a.delay_percentile[1] AS p25,
        a.delay_percentile[2] AS p50,
        a.delay_percentile[3] AS p75,
        a.delay_percentile[3] AS p90,
        b.stop_lat AS latitude,
        b.stop_lon AS longitude
    FROM {0}.sbb_orc_delays AS a
    JOIN {0}.sbb_orc_stops AS b
    ON (lower(b.stop_name)==lower(a.stop_name) OR lower(b.stop_id)==lower(a.stop_name))
    WHERE a.stop_name <> '' AND a.stop_name IS NOT NULL
    ORDER BY a.time_of_day
""".format(username)
cursor.execute(query)

# %% [markdown]
# And that's all. Executing a `SELECT * FROM sbb_orc_delays_join` will coalesce all the `VIEW` statements defined before into one big `SELECT` statement, which is then converted and run as _Map Reduce_ jobs.
#
# You can prefix the `SELECT` statement with the `EXPLAIN` or `EXPLAIN EXTENDED` and check-out the resulting _Map Reduce_ jobs - but be warned the generated code quickly becomes very hard to understand, and this is normally used by the expert mainly for debugging and optimization purposes.

# %%
cursor.execute("EXPLAIN SELECT * FROM {0}.sbb_orc_delays_join".format(username))
cursor.fetchall()

# %% [markdown]
# You can know run the query and save the results into a Pandas DataFrame.
#
# It may take a few minutes ... be patient. We highly recommend that you run this query during off-hours, when the load is light.

# %%
# %%time
df=pd.read_sql("SELECT * FROM {0}.sbb_orc_delays_join".format(username),conn)

# %%
df.info()

# %%
df.head()

# %%
# df.to_csv("./sbb_delays_join.csv",sep=';',encoding='utf-8')

# %% [markdown]
# ---
# #### 4. Map where delays happen
#
# You have now all the data you need in a panda DataFrame. The data is aggregated and its memory footprint is very reasonable. You can further analyse it locally, in this notebook.
#
# Below we display the data in an [plotly.express.density_mapbox](https://plotly.com/python-api-reference/generated/plotly.express.density_heatmap.html), one mode of transport at the time. Modify the next cell in order to filter the mode of transport.

# %%
dmode=df.loc[df['sbb_orc_delays_join.mode']=='zug'].sort_values(by='sbb_orc_delays_join.time_of_day')
dmode.describe()

# %%
import plotly.express as px
fig = px.density_mapbox(
    dmode,
    lat = 'sbb_orc_delays_join.latitude',
    lon = 'sbb_orc_delays_join.longitude',
    z = 'sbb_orc_delays_join.p50',
    animation_frame = "sbb_orc_delays_join.time_of_day",
    hover_name = 'sbb_orc_delays_join.stop_name',
    labels = {
        'sbb_orc_delays_join.p50': 'Delay (seconds)',
        'sbb_orc_delays_join.time_of_day': "15min",
        'sbb_orc_delays_join.latitude': 'Latitude',
        'sbb_orc_delays_join.longitude': 'Longitude'
    },
    title = "Daily delays",
    opacity = 0.8,
    range_color = [0,100],
    radius = 10,
    color_continuous_scale = 'Turbo', #'OrRd',
    mapbox_style = 'carto-positron',
    zoom = 10,
    width = 900,
    height = 600
)

fig.show()

# %%
