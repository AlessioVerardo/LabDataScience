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
# ---
# # Exercise notebook
#
# _À vous de jouer!_
# _Your turn!_
# _Jetzt sind dran!_
#
# In this notebook you will extend the queries seen earlier to extract a richer data set.
#
# This is the last part of a series of 3 revision notebooks about Hive. They are designed so that they can be reviewed as standalone notebooks.

# %% [markdown]
# ---
# Set up your environment, this is the same as before

# %%
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

hiveaddr = os.environ['HIVE_SERVER2']
username = os.environ['RENKU_USERNAME']
print(hiveaddr)
print(username)

# %%
from pyhive import hive

(hivehost,hiveport)=hiveaddr.split(':')
conn = hive.connect(host=hivehost,port=hiveport,username=username)
cursor = conn.cursor()

# %%
query = "CREATE DATABASE IF NOT EXISTS {0}".format(username)
cursor.execute(query)

# %%
query = "USE {0}".format(username)
cursor.execute(query)

# %%
cursor.execute('SHOW TABLES in {0}'.format(username))
cursor.fetchall()

# %%
cursor.execute('DESCRIBE {0}.sbb_orc_istdaten'.format(username))
cursor.fetchall()

# %%
import pandas as pd
pd.read_sql('SELECT * FROM {0}.sbb_orc_istdaten LIMIT 5'.format(username), conn)

# %% [markdown]
# ---
# Exercises:
#
# Can you now modify the query below so that it returns:
#
# * The interquartiles of the departure times
#     - Per category of transport, and
#     - Per stop name, and
#     - Per type of day (1 if weekend, 0 otherwise), and
#     - Per per month,
#     - Of each year
# * Aggregated hourly (instead of every 15mins)
#
# We have replaced parts of the query with TODO. You will need to recontruct the query.
#
# The table should look like:
#
# |mode|stop_name|hour|month|year|weekend|an_p10|an_p25|an_p50|an_p75|an_p90|ab_p10|ab_p25|ab_p50|ab_p75|ab_p90|latitude|longitude|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |bus|Lausanne gare|8|1|2020|1|4.55|10.0|12.0|15.0|20.0|3.55|9.0|13.1|16.45|25.3|46.5198|6.6335|

# %% [markdown]
# ---
# Use the query from the [exploration](./exploration.ipynb) notebook and create a view, in which you will replace `expected` and `actual` of the arrival times by `an_expected` and `an_actual`, and create two new columns `ab_expected` and `ab_actual`.
#
#

# %%
# %%time
cursor.execute("DROP VIEW IF EXISTS {0}.sbb_times".format(username))
query = """
CREATE VIEW {0}.sbb_times
    AS SELECT
        lower(produkt_id) AS mode,
        haltestellen_name AS stop_name,
        TODO - an_expected
        TODO - an_actual
        TODO - ab_expected
        TODO - ab_actual
        if(extract(dayofweek from CAST(unix_timestamp(ankunftszeit, 'dd.MM.yyyy HH:mm')*1000 AS TIMESTAMP))>1,0,1) as weekend,
        extract(year from CAST(unix_timestamp(ankunftszeit, 'dd.MM.yyyy HH:mm')*1000 AS TIMESTAMP)) as year,
        TODO - extract the month,
        TODO - extract the hour
    FROM {0}.sbb_orc_istdaten      
    WHERE
        produkt_id <> ''
        AND produkt_id IS NOT NULL
        AND ankunftszeit IS NOT NULL
        AND abfahrtszeit IS NOT NULL
        AND betriebstag RLIKE '^...05.2019'
        AND an_prognose_status in ('REAL', 'GESCHAETZT')
        AND haltestellen_name <> ''
        AND haltestellen_name IS NOT NULL
""".format(username)
cursor.execute(query)

# %%
# %%time
cursor.execute("DROP VIEW IF EXISTS sbb_orc_delays")
query = """
    CREATE VIEW {0}.sbb_orc_delays
    AS SELECT
        mode,
        stop_name,
        hour,
        month,
        year,
        weekend,
        TODO - extract an_delay_percentile,
        TODO - extract ab_delay_percentile
    FROM {0}.sbb_times
        TODO - complete the GROUP BY mode,stop_name,...
""".format(username)
cursor.execute(query)

# %%
# %%time
cursor.execute("DROP VIEW IF EXISTS {0}.sbb_orc_delays_join".format(username))
query = """
    CREATE VIEW {0}.sbb_orc_delays_join
    AS SELECT DISTINCT
        a.mode,
        a.stop_name,
        a.hour,
        a.an_delay_percentile[0] AS an_p10,
        a.an_delay_percentile[1] AS an_p25,
        a.an_delay_percentile[2] AS an_p50,
        a.an_delay_percentile[3] AS an_p75,
        a.an_delay_percentile[3] AS an_p90,
        a.ab_delay_percentile[0] AS ab_p10,
        a.ab_delay_percentile[1] AS ab_p25,
        a.ab_delay_percentile[2] AS ab_p50,
        a.ab_delay_percentile[3] AS ab_p75,
        a.ab_delay_percentile[3] AS ab_p90,
        b.stop_lat AS latitude,
        b.stop_lon AS longitude
    FROM {0}.sbb_orc_delays AS a
    JOIN classdb.sbb_orc_stops AS b
    ON (lower(b.stop_name)==lower(a.stop_name) OR lower(b.stop_id)==lower(a.stop_name))
    WHERE a.stop_name <> '' AND a.stop_name IS NOT NULL
""".format(username)
cursor.execute(query)

# %%
# %%time
df=pd.read_sql("SELECT * FROM {0}.sbb_orc_delays_join".format(username),conn)

# %%
