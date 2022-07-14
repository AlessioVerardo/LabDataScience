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
# # Homework 2 - Data Wrangling with Hadoop
#
# The goal of this assignment is to put into action the data wrangling techniques from the exercises of week-3 and week-4. We highly suggest you to finish these two exercises first and then start the homework. In this homework, we are going to reuse the same __sbb__ and __twitter__ datasets as seen before during these two weeks. 
#
# ## Hand-in Instructions
# - __Due: 05.04.2022 23:59 CET__
# - Fork this project as a private group project
# - Verify that all your team members are listed as group members of the project
# - `git push` your final verion to your group's Renku repository before the due date
# - Verify that `Dockerfile`, `environment.yml` and `requirements.txt` are properly written and notebook is functional
# - Add necessary comments and discussion to make your queries readable
#
# ## Hive Documentation
#
# Hive queries: <https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Select>
#
# Hive functions: <https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF>

# %% [markdown]
# <div style="font-size: 100%" class="alert alert-block alert-warning">
#     <b>Get yourself ready:</b> 
#     <br>
#     Before you jump into the questions, please first go through the notebook <a href='./prepare_env.ipynb'>prepare_env.ipynb</a> and make sure that your environment is properly set up.
#     <br><br>
#     <b>Cluster Usage:</b>
#     <br>
#     As there are many of you working with the cluster, we encourage you to prototype your queries on small data samples before running them on whole datasets.
#     <br><br>
#     <b>Try to use as much HiveQL as possible and avoid using pandas operations. Also, whenever possible, try to apply the methods you learned in class to optimize your queries to minimize the use of computing resources.</b>
# </div>

# %% [markdown]
# ## Part I: SBB/CFF/FFS Data (40 Points)
#
# Data source: <https://opentransportdata.swiss/en/dataset/istdaten>
#
# In this part, you will leverage Hive to perform exploratory analysis of data published by the [Open Data Platform Swiss Public Transport](https://opentransportdata.swiss).
#
# Format: the dataset is originally presented as a collection of textfiles with fields separated by ';' (semi-colon). For efficiency, the textfiles have been converted into Optimized Row Columnar ([_ORC_](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+ORC)) file format. 
#
# Location: you can find the data in ORC format on HDFS at the path `/data/sbb/part_orc/istdaten`.
#
# The full description from opentransportdata.swiss can be found in <https://opentransportdata.swiss/de/cookbook/ist-daten/> in four languages. There may be inconsistencies or missing information between the translations.. In that case we suggest you rely on the German version and use an automated translator when necessary. We will clarify if there is still anything unclear in class and Slack. Here we remind you the relevant column descriptions:
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
# __Initialization__

# %%
import os
import pandas as pd
pd.set_option("display.max_columns", 50)
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

username = os.environ['RENKU_USERNAME']
hiveaddr = os.environ['HIVE_SERVER2']
(hivehost,hiveport) = hiveaddr.split(':')
print("Operating as: {0}".format(username))

# %%
from pyhive import hive

# create connection
conn = hive.connect(host=hivehost, 
                    port=hiveport,
                    username=username) 
# create cursor
cur = conn.cursor()

# %% [markdown]
# ### a) Prepare the table - 5/40
#
# Complete the code in the cell below, replace all `TODO` in order to create a Hive Table of SBB Istadaten.
#
# The table has the following properties:
#
# * The table is in your database, which must have the same name as your gaspar ID
# * The table name is `sbb_orc`
# * The table must be external
# * The table content consist of ORC files in the HDFS folder `/data/sbb/part_orc/istdaten`
# * The table is _partitioned_, and the number of partitions should not exceed 50
#

# %%
### Create your database if it does not exist
query = """
CREATE DATABASE IF NOT EXISTS {0} LOCATION '/user/{0}/hive'
""".format(username)
cur.execute(query)

# %%
### Make your database the default
query = """
USE {0}
""".format(username)
cur.execute(query)

# %%
query = """
DROP TABLE IF EXISTS {0}.sbb_orc
""".format(username)
cur.execute(query)

# %%
query = """
CREATE EXTERNAL TABLE {0}.sbb_orc(
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
    PARTITIONED BY (year smallint, month smallint)
    ROW FORMAT DELIMITED FIELDS TERMINATED BY ';'
    STORED AS ORC
    LOCATION '/data/sbb/part_orc/istdaten'
    TBLPROPERTIES ('skip.header.line.count'='1')
""".format(username)
cur.execute(query)

# %%
query = """MSCK REPAIR TABLE {0}.sbb_orc""".format(username)
cur.execute(query)

# %%
query = """SELECT * FROM  {0}.sbb_orc LIMIT 5""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# We choose to keep all the fields in a string type for two reasons: 
# - The first one is that all the exercises are done in this way and it seems easier to use.
# - Creating the table with the "correct" type (i.e. boolean for boolean columns and date for some columns) would require a lot of pre-processing and after discussion with the teaching team, we think that it is not worth it. 
#
# Also note that we include more fields than the ones described in the introduction. The reasons for that is that the data is already available on the disk (i.e. we do not need to create it) and it will be unfortunate not to use all the information at our disposal.

# %% [markdown]
# **Checkpoint**
#
# Run the cells below and verify that your table satisfies all the required properties

# %%
query = """
DESCRIBE {0}.sbb_orc
""".format(username)
cur.execute(query)
cur.fetchall()

# %%
query = """
SHOW PARTITIONS {0}.sbb_orc
""".format(username)
cur.execute(query)
val = cur.fetchall()
val, len(val)

# %% [markdown]
# Furthermore, we can see that we have some default partition. Since we are still under the limit of 50 partitions, we choose to keep it. 
#
# Finally, the table seems to be in the correct format, i.e. all fields are available and in string format. 

# %% [markdown]
# ### b) Type of transport - 5/40
#
# In the exercise of week-3, you have already explored the stop distribution of different types of transport on a small data set. Now, let's do the same for a full two years worth of data.
#
# - Query `sbb_orc` to get the total number of stops for different types of transport in each month of 2019 and 2020, and order it by time and type of transport.
# |month_year|ttype|stops|
# |---|---|---|
# |...|...|...|
# - Use `plotly` to create a facet bar chart partitioned by the type of transportation. 
# - Document any patterns or abnormalities you can find.
#
# __Note__: 
# - In general, one entry in the `sbb_orc` table means one stop.
# - You might need to filter out the rows where:
#     - `BETRIEBSTAG` is not in the format of `__.__.____`
#     - `PRODUKT_ID` is NULL or empty
# - Facet plot with plotly: https://plotly.com/python/facet-plots/

# %%
# You may need more than one query, do not hesitate to create as many as you need.
# Change month, year to int, <> instead of !=, remove concat

query = """
SELECT CONCAT(year,'_' , substr(CONCAT('0',month),-2)) as month_year, lower(PRODUKT_ID) as ttype, COUNT(1) as stops
FROM {0}.sbb_orc 
WHERE (year == 2019 or year == 2020) 
    and (PRODUKT_ID is not NULL) 
    and (PRODUKT_ID <> '') 
    and (betriebstag LIKE '__.__.____')
GROUP BY YEAR, MONTH, lower(PRODUKT_ID)
ORDER BY YEAR, MONTH, lower(PRODUKT_ID)
""".format(username)
df_ttype = pd.read_sql(query, conn)

# %% [markdown]
# The function ```substr(CONCAT('0',CAST(month AS string)),-2)``` allows us to obtain the month number (an integer) in the 2 digits format, i.e. 1 will become 01. Therfore, the whole command ```CONCAT(CAST(year AS string),'_' ,substr(CONCAT('0',CAST(month AS string)),-2))```allows us to obtain the year and the month in the following format: yyyy_mm

# %%
# Create a month dictionary for the visualisation axis.
months = {'01':"Jan",'02':"Feb", '03':"Mar",'04':"Apr", '05':"May", '06':"Jun", '07':"Jul",'08':"Aug", '09':"Sep", '10':"Oct", '11':"Nov",'12':"Dec"}
# Create year and month columns
df_ttype['year'] = df_ttype.apply(lambda x: x['month_year'][:4],axis=1)
df_ttype['month'] = df_ttype.apply(lambda x: months[x['month_year'][-2:]],axis=1)
df_ttype.head(5)

# %%
fig = px.bar(df_ttype, x="month", y="stops", facet_row="year", facet_col="ttype",log_y=True)
fig.update_layout({
    "title":"Histogram of the number of stops by transport and by month for 2019 and 2020"
})
# Remove the axis title
fig.for_each_yaxis(lambda y: y.update(title = ''))
fig.for_each_xaxis(lambda x: x.update(title = ''))
# Add the axis in a more readable way using annotations
fig.add_annotation(x=-0.05,y=0.5,
                   text="Number of stops", textangle=-90,
                    xref="paper", yref="paper")
fig.add_annotation(x=0.5,y=-0.25,
                   text="Month of year",
                    xref="paper", yref="paper")
fig.show()

# %% [markdown]
# For visualisation purposes, we use logarithm axis for the $y$-axis. 
#
# The first interesting thing to note is that it seems that there are missing data for April and May 2020 for the ship transports. Also, data are missing for the metro transport for the first 3 months of 2019 and almost the whole 2019 year for the Rack railway (`zahnradbahn`). We think that this might be the case because the data collection starts only at this moment, since we go from 0 stop to a certain amount from month to another. 
#
# One final comment is the fact that for every transport, there is a decrease in the number of stops for July 2019, this might due to some data lost.

# %% [markdown]
# ### c) Schedule - 10/40
#
# - Select a any day on a typical week day (not Saturday, not Sunday, not a bank holiday) from `sbb_orc`. Query the table for that one-day and get the set of IC (`VERKEHRSMITTEL_TEXT`) trains you can take to go (without connections) from Genève to Lausanne on that day. 
# - Display the train number (`LINIEN_ID`) as well as the schedule (arrival and departure time) of the trains.
#
# |train_number|departure|arrival|
# |---|---|---|
# |...|...|...|
#
# __Note:__ 
# - The schedule of IC from Genève to Lausanne has not changed for the past few years. You can use the advanced search of SBB's website to check your answer.
# - Do not hesitate to create intermediary tables or views (see [_CTAS_](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTableAsSelect(CTAS)))
# - You might need to add filters on these flags: `ZUSATZFAHRT_TF`, `FAELLT_AUS_TF`, `DURCHFAHRT_TF` 
# - Functions that could be useful: `unix_timestamp`, `to_utc_timestamp`, `date_format`.

# %%
# Drop potentially existing view
cur.execute("DROP VIEW IF EXISTS {0}.view_lausanne".format(username))
cur.execute("DROP VIEW IF EXISTS {0}.view_geneve".format(username))

# %% [markdown]
# In the following, we will create two views:
# - One for all the IC going through Lausanne on April 27th 2018. Lausanne main station is written `Lausanne` in the dataset.
# - One for all the IC going thorugh Geneva on April 27th 2018. Geneva main station is written `Genève` in the dataset.
#
# To create the views, we needed to filter to have only the IC (`VERKEHRSMITTEL_TEXT`),`DURCHFAHRT_TF`=false that effectively stop in Lausanne (resp. Geneva), `FAELLT_AUS_TF`=false so that the trip is not cancelled and `ZUSATZFAHRT_TF`=false so that we have only the usual trip.

# %%
DAY = '27.04.2018'

# %%
view_lausanne = """
CREATE VIEW {0}.view_lausanne
    AS SELECT
        FAHRT_BEZEICHNER,
        LINIEN_ID,
        ankunftszeit
    FROM {0}.sbb_orc      
    WHERE ZUSATZFAHRT_TF == 'false'
    AND FAELLT_AUS_TF =='false'
    AND DURCHFAHRT_TF =='false'
    AND lower(HALTESTELLEN_NAME) like 'lausanne'
    AND BETRIEBSTAG like '{1}'
    AND lower(PRODUKT_ID) LIKE 'zug'
    AND lower(VERKEHRSMITTEL_TEXT) LIKE 'ic'
    AND ANKUNFTSZEIT IS NOT NULL
    AND ANKUNFTSZEIT <> ''
""".format(username, DAY)
cur.execute(view_lausanne)


# %%
query = """
SELECT * FROM {0}.view_lausanne LIMIT 5  
""".format(username)
laus = pd.read_sql(query, conn)
laus

# %%
view_geneve = """
CREATE VIEW {0}.view_geneve
    AS SELECT
        FAHRT_BEZEICHNER,
        LINIEN_ID,
        ABFAHRTSZEIT
    FROM {0}.sbb_orc      
    WHERE ZUSATZFAHRT_TF=='false'
    AND FAELLT_AUS_TF =='false'
    AND DURCHFAHRT_TF =='false'
    AND lower(PRODUKT_ID) LIKE 'zug'
    AND lower(VERKEHRSMITTEL_TEXT) LIKE 'ic'
    AND lower(HALTESTELLEN_NAME) like 'genève'
    AND BETRIEBSTAG like '{1}'
    AND ABFAHRTSZEIT IS NOT NULL
    AND ABFAHRTSZEIT <> ''
""".format(username, DAY)
cur.execute(view_geneve)

# %%
query = """
SELECT * FROM {0}.view_geneve LIMIT 5
""".format(username)
genf = pd.read_sql(query, conn)
genf

# %%
query = """
SELECT gen.linien_id as train_number, 
    date_format(from_unixtime(unix_timestamp(gen.ABFAHRTSZEIT,'dd.MM.yyyy HH:mm')), 'HH:mm') as Departure,
    date_format(from_unixtime(unix_timestamp(lau.ankunftszeit,'dd.MM.yyyy HH:mm')), 'HH:mm') as Arrival
FROM {0}.view_geneve gen 
INNER JOIN {0}.view_lausanne lau ON gen.FAHRT_BEZEICHNER = lau.FAHRT_BEZEICHNER AND gen.linien_id = lau.linien_id
WHERE unix_timestamp(gen.ABFAHRTSZEIT,'dd.MM.yyyy HH:mm')< unix_timestamp(lau.ankunftszeit,'dd.MM.yyyy HH:mm')
ORDER BY unix_timestamp(gen.ABFAHRTSZEIT,'dd.MM.yyyy HH:mm')
""".format(username)
ics = pd.read_sql(query, conn)

# %%
ics.head(16)

# %% [markdown]
# To get the final schedule, we need to join the two previous views. To do so, we join them on the train number (`linien_id`) and same trip identifiers (`FAHRT_BEZEICHNER`). Finally, we had to filter to get only the train starting from Lausanne (i.e. departure time from Lausanne is earlier than the arrival time at Geneve) which is the reason why we have the `WHERE` conditions.
#
# We can see that this schedule and the current SBB schedule (that can be found on the [SBB website](https://www.cff.ch) are almost identical (the train arrive 1 minute later at Geneva).

# %% [markdown]
# ### d) Delay percentiles - 10/40
#
# - Query `sbb_orc` to compute the 50th and 75th percentiles of __arrival__ delays for IC 702, 704, ..., 728, 730 (15 trains total) at Genève main station. 
# - Use `plotly` to plot your results in a proper way. 
# - Which trains are the most disrupted? Can you find the tendency and interpret?
#
# __Note:__
# - Do not hesitate to create intermediary tables. 
# - When the train is ahead of schedule, count this as a delay of 0.
# - Use only stops with `AN_PROGNOSE_STATUS` equal to __REAL__ or __GESCHAETZT__.
# - Functions that may be useful: `unix_timestamp`, `percentile_approx`, `if`

# %%
cur.execute("DROP VIEW IF EXISTS {0}.geneve_delay".format(username))
query = """
CREATE VIEW {0}.geneve_delay
    AS SELECT
        LINIEN_ID,
        unix_timestamp(ankunftszeit, 'dd.MM.yyyy HH:mm') AS an_expected, 
        unix_timestamp(an_prognose,  'dd.MM.yyyy HH:mm:ss') AS an_actual
    FROM {0}.sbb_orc      
    WHERE lower(HALTESTELLEN_NAME) like 'genève'
    AND AN_PROGNOSE_STATUS in ('REAL', 'GESCHAETZT')
    AND lower(VERKEHRSMITTEL_TEXT) like 'ic'
    AND (LINIEN_ID BETWEEN 702 and 730) AND (LINIEN_ID % 2==0)
""".format(username)
cur.execute(query)

# %% [markdown]
# We create a view with the relevant information for our queries.
# The selection for having only the IC 702, ..., 730 is done by checking that the IC number is between 702 and 730 and by checking that it is an even number (modulo 2 == 0). We also filter so that the current stop is Genève (this corresponds to the main station of Geneva).

# %%
cur.execute("DROP VIEW IF EXISTS {0}.geneve_delay_perc".format(username))
query = """
CREATE VIEW {0}.geneve_delay_perc
    AS SELECT 
    LINIEN_ID,
    percentile_approx(IF(an_actual > an_expected, an_actual - an_expected, 0),ARRAY(0.5,0.75)) as an_delay_percentile
FROM {0}.geneve_delay
GROUP BY LINIEN_ID
""".format(username)
cur.execute(query)


# %% [markdown]
# Here, we computed the approximate percentile (as the delay are float) for each train, based on the data in the previously created view.

# %%
query = """
SELECT
    CONCAT('IC', LINIEN_ID) as name,
    an_delay_percentile[0] as an_50,
    an_delay_percentile[1] as an_75
FROM {0}.geneve_delay_perc
ORDER BY LINIEN_ID
""".format(username)
df_delays_ic_gen = pd.read_sql(query, conn)

# %% [markdown]
# Finally, we simply select the name and the delay percentile to get our actual data.  

# %%
df_delays_ic_gen.head(5)

# %%
# Reformat and rename the dataframe so that it is easier to display the information
df_delays_ic_gen = df_delays_ic_gen.rename(columns={"an_50":"50th", "an_75":"75th", "name":"Train"})
df_delays_ic_gen = df_delays_ic_gen.melt(id_vars=['Train'], value_vars=['50th','75th'], value_name="Delay", var_name="Percentile")

# %%
fig = px.bar(
    df_delays_ic_gen, 
    x='Train',
    y='Delay',
    color='Percentile',
    barmode="group"
)

fig.update_layout(
{
    "title":"Delay by IC train",
    "yaxis_title":"Delay (sec)",
})

fig.show()

# %% [markdown]
# After investigation, we found that the LINIEN_ID is in fact a number depending on a specific line and the time at which the train is departing. This means that IC702 is always going to do the same trip at the same hour. Furthermore, from point 1.c, we can see that this seems to be confirmed as the IC number is increasing as the hour of the day increases. Indeed, it is possible show (by inverting all the queries done in 1c) that the IC702 do the trip Bern -> Lausanne -> Geneva at 5:34am, IC704 do the trip Bern -> Lausanne -> Geneva at 6:34am, etc....
#
# From this, we can see that the most delayed train are IC 706, 708, 710, 730 corresponds to the rush hours (i.e. people going or returning from work) and it makes sense that the train will have a bit of delay at this time. 

# %% [markdown]
# ### e) Delay heatmap 10/40
#
# - For each week (1 to 52) of each year from 2019 to 2021, query `sbb_orc` to compute the median of delays of all trains __departing__ from any train stations in Zürich area during that week. 
# - Use `plotly` to draw a heatmap year x week (year columns x week rows) of the median delays. 
# - In which weeks were the trains delayed the most/least? Can you explain the results?
#
# __Note:__
# - Do not hesitate to create intermediary tables. 
# - When the train is ahead of schedule, count this as a delay of 0 (no negative delays).
# - Use only stops with `AB_PROGNOSE_STATUS` equal to __REAL__ or __GESCHAETZT__.
# - For simplicty, a train station in Zürich area <=> it's a train station & its `HALTESTELLEN_NAME` starts with __Zürich__.
# - Heatmap with `plotly`: https://plotly.com/python/heatmaps/
# - Functions that may be useful: `unix_timestamp`, `from_unixtime`, `weekofyear`, `percentile_approx`, `if`

# %% [markdown]
# The first query below will select all the relevant data for our task. All the filters are given in the statement. Note that we also filter by `ABFAHRTSZEIT` not being null as we are interested by the train arriving at Zurich and not departing from Zurich. 

# %%
cur.execute("DROP VIEW IF EXISTS {0}.zurich_zug".format(username))

query = """
CREATE VIEW {0}.zurich_zug
    AS SELECT
    unix_timestamp(abfahrtszeit, 'dd.MM.yyyy HH:mm') AS ab_expected, 
    unix_timestamp(ab_prognose,  'dd.MM.yyyy HH:mm:ss') AS ab_actual,
    year,
    weekofyear(from_unixtime(unix_timestamp(abfahrtszeit, 'dd.MM.yyyy HH:mm'),'yyyy-MM-dd')) as week
    FROM {0}.sbb_orc
WHERE lower(HALTESTELLEN_NAME) like 'zürich%'
    AND lower(PRODUKT_ID) like 'zug'
    AND AB_PROGNOSE_STATUS in ('REAL', 'GESCHAETZT')
    AND (year between 2019 and 2021)
    AND ABFAHRTSZEIT is not NULL
    AND ABFAHRTSZEIT <> ''
""".format(username)
cur.execute(query)

# %% [markdown]
# Compute the 50th percentile for each week of each year as before:

# %%
query = """
    SELECT 
    year,
    week,
    percentile_approx(IF(ab_actual > ab_expected, ab_actual - ab_expected, 0),0.5) as delay_median
FROM {0}.zurich_zug
GROUP BY year, week
ORDER BY year, week
""".format(username)
cur.execute(query)
df_delays_zurich = pd.read_sql(query, conn)

# %%
len(df_delays_zurich[df_delays_zurich['year']==2021])

# %% [markdown]
# By looking at the number of line in the 2021 dataframe, we can see that we have 45 instead of 52 weeks for the year 2021 so it seems that we have missing data. We will now take a look at the HDFS data to see if the data are indeed missing or if we made a mistake in the query. We start by selecting all the data for year 2021. 

# %%
query = """
SELECT
    unix_timestamp(abfahrtszeit, 'dd.MM.yyyy HH:mm') AS ab_expected, 
    unix_timestamp(ab_prognose,  'dd.MM.yyyy HH:mm:ss') AS ab_actual,
    year
    FROM {0}.sbb_orc
WHERE HALTESTELLEN_NAME like 'Zürich%'
    AND PRODUKT_ID like 'Zug'
    AND AN_PROGNOSE_STATUS in ('REAL', 'GESCHAETZT')
    AND year == 2021
ORDER BY abfahrtszeit
""".format(username)
val = pd.read_sql(query, conn)

# %%
# Transform the timestamp into human understandable date
val['ab_expected'] =pd.to_datetime(val['ab_expected'], unit='s') 
val['ab_actual'] =pd.to_datetime(val['ab_actual'], unit='s') 


# %%
val[val['ab_expected']<'2021-03-01']

# %% [markdown]
# As explained on Slack and what we've just seen on the above cell, we can see that the only available data from January and February 2021 are from January 25th to January 28th. So, we can assume that our query is correct and that there are simply missing data for this part of the year.
#
# Therefore, we will need to fill the data with N/A in order to have a square heatmap, i.e. for the weeks without any data, we will put N/A. 

# %%
new_df = df_delays_zurich.set_index(['year','week']).unstack('year')
df_to_array = new_df.to_numpy()
new_df.head()

# %%
# Create the heatmap, put values with custom axis values.
fig = px.imshow(
    df_to_array,
    y=list(range(1,54)),
    x=list(range(2019,2022)),
    labels=dict(y="Week of year", x="Year", color="Delay median (sec)"),
    width=700,
    height=700
)
fig.update_xaxes(side="top")
fig.update_layout(
    title = {
        'text': "Heatmap of train delays medians from 2019 to 2021",
        'x':0.6,
        'y': 0.05,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title = "Week of year",
    yaxis_title = "Year"
)

fig.show()

# %% [markdown]
# Note that the NaN values correspond to missing data in the dataset (previously added data). Since for one of the years, we got data for week 53, we chose to keep it. It seems that trains encounter the largest delays around week 44 (corresponding to the beginning of November), maybe due to the starting of winter and snowfalls which can disrupt trains. And one can see that the trains with the smallest delays occurred around weeks 12-20 of 2020 which correspond to the midst of the covid outbreak and lockdowns, which meant people were less likely to travel (to work or other...). And less people translate into less delays in general.

# %% [markdown]
# ## Part II: Twitter Data (20 Points)
#
# Data source: https://archive.org/details/twitterstream?sort=-publicdate 
#
# In this part, you will leverage Hive to extract the hashtags from the source data, and then perform light exploration of the prepared data. 
#
# ### Dataset Description 
#
# Format: the dataset is presented as a collection of textfiles containing one JSON document per line. The data is organized in a hierarchy of folders, with one file per minute. The textfiles have been compressed using bzip2. In this part, we will mainly focus on __2016 twitter data__.
#
# Location: you can find the data on HDFS at the path `/data/twitter/json/year={year}/month={month}/day={day}/{hour}/{minute}.json.bz2`. 
#
# Relevant fields: 
# - `created_at`, `timestamp_ms`: The first is a human-readable string representation of when the tweet was posted. The latter represents the same instant as a timestamp since UNIX epoch.
# - `lang`: the language of the tweet content 
# - `entities`: parsed entities from the tweet, e.g. hashtags, user mentions, URLs.
# - In this repository, you can find [a tweet example](../data/tweet-example.json).
#
# Note:  Pay attention to the time units! and check the Date-Time functions in the Hive [_UDF_](https://cwiki.apache.org/confluence/display/hive/Languagemanual+udf#LanguageManualUDF-DateFunctions) documentation.

# %% [markdown]
# <div style="font-size: 100%" class="alert alert-block alert-danger">
#     <b>Disclaimer</b>
#     <br>
#     This dataset contains unfiltered data from Twitter. As such, you may be exposed to tweets/hashtags containing vulgarities, references to sexual acts, drug usage, etc.
#     </div>

# %% [markdown]
# ### a) JsonSerDe - 4/20
#
# In the exercise of week 4, you have already seen how to use the [SerDe framework](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-RowFormats&SerDe) to extract JSON fields from raw text format. 
#
# In this question, please use SerDe to create an <font color="red" size="3px">EXTERNAL</font> table with __one day__ (e.g. 01.07.2016) of twitter data. You only need to extract three columns: `timestamp_ms`, `lang` and `entities`(with the field `hashtags` only) with following schema (you need to figure out what to fill in `TODO`):
# ```
# timestamp_ms string,
# lang         string,
# entities     struct<hashtags:array<...<text:..., indices:...>>>
# ```
#
# The table you create should be similar to:
#
# | timestamp_ms | lang | entities |
# |---|---|---|
# | 1234567890001 | en | {"hashtags":[]} |
# | 1234567890002 | fr | {"hashtags":[{"text":"hashtag1","indices":[10]}]} |
# | 1234567890002 | jp | {"hashtags":[{"text":"hashtag1","indices":[14,23]}, {"text":"hashtag2","indices":[45]}]} |
#
# __Note:__
#    - JsonSerDe: https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-RowFormats&SerDe
#    - Hive data types: https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Types#LanguageManualTypes
#    - Hive complex types: https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Types#LanguageManualTypes-ComplexTypes

# %%
# Create a table with all the Twitter data, with the corresponding partitions
# using deserialization for json strings

query="""
    DROP TABLE IF EXISTS {0}.twitter_hashtags
""".format(username)
cur.execute(query)

query="""
    CREATE EXTERNAL TABLE {0}.twitter_hashtags(
    timestamp_ms STRING,
    lang STRING,
    entities STRUCT<hashtags:ARRAY<STRUCT<text:STRING, indices:ARRAY<Int>>>>
    )
    PARTITIONED BY (year STRING, month STRING, day STRING)
    ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
    WITH SERDEPROPERTIES(
        "ignore.malformed.json"="true"
    )
    STORED AS TEXTFILE
    LOCATION '/data/twitter/json/'

""".format(username)
cur.execute(query)

# %%
# Build the partitions on years, months, days

query="""
    MSCK REPAIR TABLE {0}.twitter_hashtags
""".format(username)
cur.execute(query)

# %%
# Select data for a single day, making use of partitions

query="""
    SELECT
    timestamp_ms, lang, entities
    FROM {0}.twitter_hashtags
    WHERE
    year=2016 AND month=07 AND day=01 --AND timestamp_ms IS NOT NULL --if we want to drop the None rows
    LIMIT 10
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# Note that we don't drop the None rows here as this query is only to inspect the content and "appearance" of the table. Also note that since the rows are not in a particular order, some have no hashtags.

# %% [markdown]
# ### b) Explosion - 4/20
#
# In a), you created a table where each row could contain a list of multiple hashtags. Create another table **containing one day of data only** by normalizing the table obtained from the previous step. This means that each row should contain exactly one hashtag. Include `timestamp_ms` and `lang` in the resulting table, as shown below.
#
# | timestamp_ms | lang | hashtag |
# |---|---|---|
# | 1234567890001 | es | hashtag1 |
# | 1234567890001 | es | hashtag2 |
# | 1234567890002 | en | hashtag2 |
# | 1234567890003 | zh | hashtag3 |
#
# __Note:__
#    - `LateralView`: https://cwiki.apache.org/confluence/display/Hive/LanguageManual+LateralView
#    - `explode` function: <https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-explode>

# %%
query="""
    DROP TABLE IF EXISTS {0}.twitter_hashtags_norm
""".format(username)
cur.execute(query)

query="""
    CREATE EXTERNAL TABLE IF NOT EXISTS {0}.twitter_hashtags_norm
    STORED AS ORC
    AS SELECT
        timestamp_ms,
        lang,
        hashtag.text AS hashtag
    FROM {0}.twitter_hashtags LATERAL VIEW explode(entities.hashtags) exploded_table AS hashtag
    WHERE year=2016 AND month=07 AND day=01
        
""".format(username)
cur.execute(query)

# %% [markdown]
# Here we split each row from previous table (twitter_hashtags) into potentially multiple rows, one for each hashtag. For this, we use the LATERAL VIEW instruction on the exploded array of hashtags.
#
# Below, we simply check that the table has the correct shape.

# %%
query="""
    SELECT * FROM {0}.twitter_hashtags_norm ORDER BY timestamp_ms LIMIT 10
""".format(username)
pd.read_sql(query, conn)

# %% [markdown]
# ### c) Hashtags - 8/20
#
# Query the normailized table you obtained in b). Create a table of the top 20 most mentioned hashtags with the contribution of each language. And, for each hashtag, order languages by their contributions. You should have a table similar to:
#
# |hashtag|lang|lang_count|total_count|
# |---|---|---|---|
# |hashtag_1|en|2000|3500|
# |hashtag_1|fr|1000|3500|
# |hashtag_1|jp|500|3500|
# |hashtag_2|te|500|500|
#
# Use `plotly` to create a stacked bar chart to show the results.
#
# __Note:__ to properly order the bars, you may need:
# ```python
# fig.update_layout(xaxis_categoryorder = 'total descending')
# ```

# %%
# You may need more than one query, do not hesitate to create more
# Number of distinct hashtags to keep in the final table

# Fix number of distinct hashtags to keep in the table.
nb = 20

query_select_lang_count = """SELECT 
        hashtag,
        lang,
        COUNT(1) AS lang_count
    FROM {0}.twitter_hashtags_norm
    GROUP BY hashtag, lang""".format(username)

query_select_total_count = """SELECT
        hashtag,
        COUNT(1) AS total_count
    FROM {0}.twitter_hashtags_norm
    GROUP BY hashtag
    ORDER BY total_count DESC
    LIMIT {1}""".format(username, nb)

query="""
    DROP TABLE IF EXISTS {0}.twitter_hashtags_counts
""".format(username)
cur.execute(query)

query = """
    CREATE EXTERNAL TABLE {0}.twitter_hashtags_counts
    STORED AS ORC
    AS SELECT
        a.hashtag,
        lang,
        lang_count,
        total_count
    FROM ({1}) AS a
    RIGHT JOIN ({2}) AS b
    ON a.hashtag == b.hashtag
    WHERE a.hashtag IS NOT NULL
    ORDER BY total_count DESC, lang_count DESC
""".format(username, query_select_lang_count, query_select_total_count)
cur.execute(query)

# %% [markdown]
# Here we assume that we need to keep only 20 distinct hashtags in the table (and not more). In the cell above, we have stored subqueries in string variables. This has 3 advantages :
# - Avoids to create multiple "intermediate" table that we won't use except for creating the final table;
# - Will execute the creation of the final table in a "single transaction" as the selects operations are just subqueries executed during the creation of the table;
# - This is far more readable compared to a query with multiple subqueries in it.
#     
# Here also note that we filter on the hashtag and we do not keep it if it is NULL. We limit the number of distinct hashtags in the second subquery where we compute the total count per hashtag. This is in turn reflected by the right join to create the final table. Note that the actual table might have more than 20 rows but it will only have 20 distinct hashtags (multiple rows can have the same hashtag if it appeared in different languages). Finally, we order them first by total count and internally by language count.

# %%
query="""
    
    SELECT * FROM {0}.twitter_hashtags_counts
    
""".format(username)
df_hashtag = pd.read_sql(query, conn)

# %%
# Check the shape of the dataframe obtained
df_hashtag

# %%
# Verify we only have 20 distinct hashtags
df_hashtag['twitter_hashtags_counts.hashtag'].nunique()

# %%
# Rename the columns to drop the table name
df_hashtag.rename(columns=lambda col: col.split('.')[1], inplace=True)

# %%
# Make the stacked bar plot
fig = px.bar(
    df_hashtag,
    x='hashtag',
    y='lang_count',
    color='lang',
    color_discrete_sequence=px.colors.qualitative.Dark24
)
# change color palette
fig.update_layout(
    xaxis_categoryorder = 'total descending',
    title='Counts of tags by language',
    xaxis_title='Hashtag',
    yaxis_title='Count'
)

fig.show()

# %% [markdown]
# In the plot above, we can first observe that the predominant language is unsuprisingly English. There are also a large gap between the first and second hashtags with the most total counts. However, if disable the English language the ordering changes a bit, the first and second swapped their positions. It is also interesting to see that some tags have very diverse languages (e.g. EURO2016, which is something international) whereas some have very few different ones with a largely dominant one, especially once english is disabled.

# %% [markdown]
# ### d) HBase - 4/20
#
# In the lecture and exercise of week-4, you have learnt what's HBase, how to create an Hbase table and how to create an external Hive table on top of the HBase table. Now, let's try to save the results of question c) into HBase, such that each entry looks like:
# ```
# (b'PIE', {b'cf1:total_count': b'31415926', b'cf2:langs': b'ja,en,ko,fr'})
# ``` 
# where the key is the hashtag, `total_count` is the total count of the hashtag, and `langs` is a string of  unique language abbreviations concatenated with commas. 
#
# __Note:__
# - To accomplish the task, you need to follow these steps:
#     - Create an Hbase table called `twitter_hbase`, in **your hbase namespace**, with two column families and fields (cf1, cf2)
#     - Create an external Hive table called `twitter_hive_on_hbase` on top of the Hbase table. 
#     - Populate the HBase table with the results of question c).
# - You may find function `concat_ws` and `collect_list` useful.

# %%
# Imports and basic setup for hbase

import happybase
hbaseaddr = os.environ['HBASE_SERVER']
hbase_connection = happybase.Connection(hbaseaddr, transport='framed',protocol='compact')

# %%
# Just drop the table if it already exists

try:
    hbase_connection.delete_table('{0}:twitter_hbase'.format(username),disable=True)
except Exception as e:
    print(e.message)
    pass

# %%
# Create the hbase table with 2 columns families

hbase_connection.create_table(
    '{0}:twitter_hbase'.format(username),
    {'cf1': dict(),
     'cf2': dict()
    }
)

# %%
# Create a Hive table, based on the content of
# our new hbase table

query="""
    DROP TABLE IF EXISTS {0}.twitter_hive_on_hbase
""".format(username)
cur.execute(query)

query = """
    CREATE EXTERNAL TABLE {0}.twitter_hive_on_hbase(
        hashtag STRING,
        total_count INT,
        langs STRING
    ) 
    STORED BY 'org.apache.hadoop.hive.hbase.HBaseStorageHandler'
    WITH SERDEPROPERTIES (
        "hbase.columns.mapping"=":key,cf1:total_count,cf2:langs"
    )
    TBLPROPERTIES(
        "hbase.table.name"="{0}:twitter_hbase",
        "hbase.mapred.output.outputtable"="{0}:twitter_hbase"
    )
""".format(username)
cur.execute(query)

# %%
# Insert data from the Hive twitter counts table created in part 2.c)

query="""
INSERT OVERWRITE TABLE {0}.twitter_hive_on_hbase
    SELECT
        hashtag,
        total_count, -- can just take min of the group since all rows for a given hashtag have the same total count
        concat_ws(',', collect_list(lang))
    FROM {0}.twitter_hashtags_counts
    GROUP BY hashtag, total_count
""".format(username)
cur.execute(query)

# %% [markdown]
# We use the propose Hive functions to first collect the languages of a hashtag as a list (after the ```GROUP BY hashtag``` clause) and then use the ```concat_ws``` to create a ',' separated string out of the elements of this list.
#
# Below, we simply scan the Hbase table to confirm the data were correctly added through the Hive table based on this Hbase table.

# %%
for r in hbase_connection.table('{0}:twitter_hbase'.format(username)).scan():
    print(r)

# %% [markdown]
# # That's all, folks!
