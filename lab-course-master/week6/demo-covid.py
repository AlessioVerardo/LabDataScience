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

# %% [markdown] slideshow={"slide_type": "slide"}
# # `DataFrame` Demo (Covid-19)

# %% [markdown]
# In this notebook we are going to create a `DataFrame` from data hosted in a [Renkulab project](https://renkulab.io/projects/covid-19/covid-19-public-data) that aggregates public data sources to better understand the spread and effect of covid-19.
# The data is refreshed every day. If you are interested, feel free to contribute by adding data sources or trying out some data science! 

# %% [markdown]
# We start a Spark application (using sparkmagic).

# %%
# %load_ext sparkmagic.magics

# %% slideshow={"slide_type": "slide"}
import os
username = os.environ['RENKU_USERNAME']
server = "http://iccluster029.iccluster.epfl.ch:8998"
from IPython import get_ipython
get_ipython().run_cell_magic('spark', line="config", 
                    cell="""{{ "name":"{0}-demo2",
                               "executorMemory":"4G",
                               "executorCores":4,
                               "numExecutors":10 }}""".format(username))

# %%
get_ipython().run_line_magic(
    "spark", "add -s {0}-demo2 -l python -u {1} -k".format(username, server)
)

# %% [markdown]
# Import and create the `SQLContext`.

# %% language="spark"
# from pyspark.sql import Row
# from pyspark.sql import SQLContext
#
# sqlContext = SQLContext(sc)

# %% [markdown]
# We copy locally a CSV file containing the confirmed cases by country and date. The source of the data is taken from the [COVID-19 data repository by John Hopkings University](https://github.com/CSSEGISandData/COVID-19).

# %% [markdown]
# Now we can create the `DataFrame` by reading the csv file, notice that the `Schema` will be inferred automatically from the first row.

# %% language="spark"
# testdf = sqlContext.read.csv("/data/covid/time_series_covid19_confirmed_global.csv",header=True, inferSchema=True)

# %% language="spark"
# testdf.printSchema()

# %% [markdown] slideshow={"slide_type": "slide"}
# Now that we know how the `schema` looks like, let's take a peek to the data.

# %% language="spark"
# testdf.select('Province/State', 'Country/Region','12/13/20').show(5)

# %% [markdown] slideshow={"slide_type": "slide"}
# To be able to perform some actions to the `DataFrame` more easily we can rename the columns of our interest.

# %% language="spark"
# testdf = testdf.withColumnRenamed("Province/State", "province_state")
# testdf = testdf.withColumnRenamed("Country/Region", "country_region")
# testdf = testdf.withColumnRenamed("12/13/20","december13")

# %% language="spark"
# testdf.printSchema()

# %% language="spark"
# testdf.select('province_state', 'country_region', 'december13').show(5)

# %% [markdown]
# Let's check the granularity of the data for Switzerland.

# %% language="spark"
# # Applying a filter by country
# testdf.filter(testdf.country_region == "Switzerland").select('province_state', 'country_region', 'december13').show()

# %% [markdown]
# Other countries have more detailed information.

# %% language="spark"
# testdf.filter(testdf.country_region == "Australia").select('province_state', 'country_region', 'december13').show()

# %% [markdown] slideshow={"slide_type": "slide"}
# We can apply an `orderBy` to the `DataFrame` to display the places with the highest number of cases.

# %% language="spark"
# testdf.orderBy(testdf.december13.desc()).select('province_state', 'country_region', 'december13').show(10)

# %%

# %%
