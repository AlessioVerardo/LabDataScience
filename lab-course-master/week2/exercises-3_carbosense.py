# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## 1. Preparing your environment

# %% [markdown]
# Run this command below if the data shows a text git lfs pointer.
# You only need to do it once when you start a new environment.

# %%
# !git lfs pull

# %% [markdown]
# Python packaged used for this exercise
# * Pandas
# * Numpy
# * Plotly, high level visualization package

# %%
import pandas as pd
import numpy as np
import plotly.express as px

# %% [markdown]
# ----
# ## 2. Visualizing Carbosense data
#
# ### 2.1 Loading the data
#
# This data contains temporal series of $CO_2$ sensor measurements in Switzerland collected by the [carbosense project](http://www.carbosense.ch/).
# * Read this data inside a python dataframe _df_.
# * Select only the meta-data of the lat,lon,alt location, the sensor (location) name, the zone area (in Zurich) of the sensors inside a dataframe _df_sensor_, making sure to drop duplicate so that we only keep one meta-data row per sensor

# %%
DATA = '../data/carbosense/CarboSense-October2017_curated.csv'

# %%
df = pd.read_csv(DATA, parse_dates=[0])
df.info()

# %% [markdown]
# #### brief data description:
#
# `Timestamp`: from 2017-10-01 to 2017-10-31
#
# `Sensor_ID`: id of the sensor
#
# `Location_Name`: name of the sensor
#
# `Zone_in_Zurich`: which zone of Zurich the sensor is located
#
# `lon` & `lat` & `altitude`: geo information of the sensor
#
# `T`: temperature
#
# `H`: humidity
#
# `CO2`: CO2
#

# %%
df.head()

# %%
df_sensor = df[['Location_Name', 'lon', 'lat', 'altitude', 'Zone_in_Zurich']].drop_duplicates('Location_Name')
df_sensor.info()

# %%
df_sensor.head()

# %% [markdown]
# ### 2.2 Aggregations

# %% [markdown]
#
#
#
# Create a new pandas DataFrame _df_co2minmax_ from _df_ and print
# the min and max _CO2_ values in each zone of `Zone_in_Zurich`
#
# Method:
# * `groupby(['Zone_in_Zurich'])`: group by zone within Zürich
# * `agg({'CO2':[min, max]})`: aggregate min and max over the _CO2_ colonne for each group
#
# References:
# [Plotly bar-charts](https://plotly.com/python/bar-charts/)

# %%
df_co2minmax=df.groupby('Zone_in_Zurich').agg({'CO2':[min,max]}).reset_index()
df_co2minmax.info()

# %% [markdown]
# * Drop the first level index _CO2_, so that the columns are called min, max instead of (_CO2_,min), (_CO2_,max)

# %%
df_co2minmax.columns=['Zone_in_Zurich','min','max']
df_co2minmax

# %% [markdown]
# * _Melt_ the DataFrame, using the method
# [pandas.DataFrame.melt](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.melt.html). This  method
# transform a wide table with many colonnes (corresponding to variable names), in a narrow (but longer) table which contains a new column containing the variable names.
# This is the inverse operation of pivot (see also [pandas.stack](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html)).
#
# <!-- 
#     - `id_vars='Zone_in_Zurich'`: column (variable) used as index around which the melt operation is executed.
#     - `var_name='minmax'`: new column that will contain the variables names (min ou max) after the _melt_
#     - `value_name='CO2'`: new column that will contain the corresponding values of the variables (values of columns min ou max in original table). -->
#
# Example:
#
# |Zone_in_Zurich|minmax|CO2|
# |--------------|------|---|
# |1|min|396.540924|
# |2|min|314.311471|
# |...|...|...|
# |1|max|557.486368|
# |2|max|3244.808898|
# |...|...|...|

# %%
## TODO: use melt to create a dataframe similar to the one above

df_co2minmax=  # Insert code here

# %% [markdown]
# * `px.bar`: Display a barplot using [plotly.express.bar](https://plotly.com/python/bar-charts/)
# * modify the axis of the figure returned by `plotly.bar`.
#     - `update_xaxes(type='category')`:  horizontal axis contains the categories

# %%
## TODO: plot min and max with barplot with barmode "group"

# Insert code here

# l'axe horizontal contient des catégories
fig.update_xaxes(type='category')
fig.show()

# %% [markdown]
# -----
# ### 2.3 Time Series
#
#
# #### 2.3.1 Visualize the time séries of a few sensors
# https://plotly.com/python/time-series/
#
# Create a dataframe _df_selection_ using the sensor subset _['ZGHD','ZORL','ZTBN']_ of _df.Location_Name_.

# %%
df_selection = df[df.Location_Name.isin(['ZGHD','ZORL','ZTBN','UTLI'])]

# %% [markdown]
# Visualisz dataframe _df_selection_.
#
# Method:
# * `px.line`: Display a line using [plotly.express.line](https://plotly.com/python/line-charts/)
#     - `x='Timestamp'`: use date and tie for horizontal axis
#     - `y='CO2'`: use variable CO2 for vertical axis
#     - `color='Location_Name'`: use different colors for each sensor

# %%
px.line(df_selection,x='Timestamp',y='CO2',color='Location_Name')

# %% [markdown]
# #### A few more decorations
# * `px.line`: plotly.express.line
#     - `labels={'Timestamp':'Date','CO2':'CO2 (ppm)','Location_Name':'Sensor'}`: dictionary of labels used for the time series
#     - `title='CO2 level in October 2017'`: Figure title
#
# * `fig.update_xaxes`: modify axis parameters
#     - `dtick=24*60*60*1000`: period of one, this is one day worth of milliseconds
#     - `ticklabelmode='period'`: label is displayed in the center of the period
#
#
# * `fig.update_layout`: modify the figure layout
#     - `hovermode="x unified"`: information displayed under the mouse pointer - other useful options are: [x, y, y unified, closest, False](https://plotly.com/python/reference/layout/#layout-hovermode)
#

# %%
## TODO: add some decorations to the vanilla figure to make it more user-friendly

# Insert code here


fig.show()

# %% [markdown]
# ### 2.3.2 With `ipywidgets` - visualize selected sensor
#
# Now it's time to add some widgets to our plot with the package `ipywidgets`. Here is [a general introduction](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Basics.html) to simple widgets in the package.

# %%
from ipywidgets import interactive, widgets, interact


# %%
def plot_co2(locations):
    
    if not locations:
        return
    
    ## TODO: copy some plotting codes from 2.3.1
    
    
    
    
    
    fig.show()

# %%
## TODO: find the widget which allows you to select multiple sensors and make an interactive plotly-based plot

# Insert code here

_ = interact(plot_co2, locations = location_selector)

# %% [markdown]
# -----
# ### 2.4. Real-time Display using (pseudo) Streaming data
#
# https://plotly.com/python/chart-events/

# %%
import plotly.graph_objects as go
import time

# %%
df_zghd = df[df.Location_Name == 'ZGHD']
# df_zghd.Timestamp = pd.to_datetime(df_zghd.Timestamp)
df_zghd.head()

# %%
# Display first day as initial historic
history = df_zghd[df_zghd.Timestamp.dt.day == 1]
# Display remaining data in real-time
stream = df_zghd[df_zghd.Timestamp.dt.day > 1].itertuples() # iterate over DataFrame rows as named tuples

# %%
# Display the historical values of the 1st day
base_fig = px.line(history, 
                   x="Timestamp", y="CO2", 
                   labels={
                          "Timestamp": "Date",
                          "CO2": "CO2 Level (ppm)"
                   },
                   title="CO2 Level in October 2017")
base_fig = base_fig.update_xaxes(
    dtick= 30 * 60 * 1000, # 30 minutes, in milliseconds units
)
base_fig = base_fig.update_layout(
    hovermode="x unified" # display the info grouped for each displacement along x
)

# %%
# Convert it to FigureWidget
fig = go.FigureWidget(base_fig)
fig
# It is also possible to `right-click` the cell and select `Create New View for Output`

# %% [markdown]
# The next cell will keep running until the month of October 2017 is fully displayed, or until you stop the Kernel.

# %%
## TODO: read char-events documentation and make a "fake" live plot

# Update data
data = fig.data[0]

try:
    for r in stream:
        
        ## TODO   
        # Send daily updates in batch, keep only last 48 values, that is 1 day (a value each 30min)
        with fig.batch_update(): 
            # Insert code here
        
        time.sleep(0.5)
        
except KeyboardInterrupt:
    pass

# %%
