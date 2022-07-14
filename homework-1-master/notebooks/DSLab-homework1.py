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
# # DSLab Homework 1 - Data Science with CO2
#
# ## Hand-in Instructions
#
# - __Due: 22.03.2022 23h59 CET__
# - `git push` your final verion to the master branch of your group's Renku repository before the due
# - check if `Dockerfile`, `environment.yml` and `requirements.txt` are properly written
# - add necessary comments and discussion to make your codes readable

# %%
# !git lfs pull

# %% [markdown]
# **We didn't use any other libraries than the ones already installed in the docker file. That's why we only specify the causal impact library as requirements and not the other libraries.**

# %% [markdown]
# ## Carbosense
#
# The project Carbosense establishes a uniquely dense CO2 sensor network across Switzerland to provide near-real time information on man-made emissions and CO2 uptake by the biosphere. The main goal of the project is to improve the understanding of the small-scale CO2 fluxes in Switzerland and concurrently to contribute to a better top-down quantification of the Swiss CO2 emissions. The Carbosense network has a spatial focus on the City of Zurich where more than 50 sensors are deployed. Network operations started in July 2017.
#
# <img src="http://carbosense.wdfiles.com/local--files/main:project/CarboSense_MAP_20191113_LowRes.jpg" width="500">
#
# <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_ZLMT_3.JPG" width="156">  <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_sensor_SMALL.jpg" width="300">

# %% [markdown]
# ## Description of the homework
#
# In this homework, we will curate a set of **CO2 measurements**, measured from cheap but inaccurate sensors, that have been deployed in the city of Zurich from the Carbosense project. The goal of the exercise is twofold: 
#
# 1. Learn how to deal with real world sensor timeseries data, and organize them efficiently using python dataframes.
#
# 2. Apply data science tools to model the measurements, and use the learned model to process them (e.g., detect drifts in the sensor measurements). 
#
# The sensor network consists of 46 sites, located in different parts of the city. Each site contains three different sensors measuring (a) **CO2 concentration**, (b) **temperature**, and (c) **humidity**. Beside these measurements, we have the following additional information that can be used to process the measurements: 
#
# 1. The **altitude** at which the CO2 sensor is located, and the GPS coordinates (latitude, longitude).
#
# 2. A clustering of the city of Zurich in 17 different city **zones** and the zone in which the sensor belongs to. Some characteristic zones are industrial area, residential area, forest, glacier, lake, etc.
#
# ## Prior knowledge
#
# The average value of the CO2 in a city is approximately 400 ppm. However, the exact measurement in each site depends on parameters such as the temperature, the humidity, the altitude, and the level of traffic around the site. For example, sensors positioned in high altitude (mountains, forests), are expected to have a much lower and uniform level of CO2 than sensors that are positioned in a business area with much higher traffic activity. Moreover, we know that there is a strong dependence of the CO2 measurements, on temperature and humidity.
#
# Given this knowledge, you are asked to define an algorithm that curates the data, by detecting and removing potential drifts. **The algorithm should be based on the fact that sensors in similar conditions are expected to have similar measurements.** 
#
# ## To start with
#
# The following csv files in the `../data/carbosense-raw/` folder will be needed: 
#
# 1. `CO2_sensor_measurements.csv`
#     
#    __Description__: It contains the CO2 measurements `CO2`, the name of the site `LocationName`, a unique sensor identifier `SensorUnit_ID`, and the time instance in which the measurement was taken `timestamp`.
#     
# 2. `temperature_humidity.csv`
#
#    __Description__: It contains the temperature and the humidity measurements for each sensor identifier, at each timestamp `Timestamp`. For each `SensorUnit_ID`, the temperature and the humidity can be found in the corresponding columns of the dataframe `{SensorUnit_ID}.temperature`, `{SensorUnit_ID}.humidity`.
#     
# 3. `sensor_metadata_updated.csv`
#
#    __Description__: It contains the name of the site `LocationName`, the zone index `zone`, the altitude in meters `altitude`, the longitude `LON`, and the latitude `LAT`. 
#
# Import the following python packages:

# %%
import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt

# %%
pd.options.mode.chained_assignment = None

# %% [markdown]
# ## PART I: Handling time series with pandas (10 points)

# %% [markdown]
# ### a) **8/10**
#
# Merge the `CO2_sensor_measurements.csv`, `temperature_humidity.csv`, and `sensors_metadata_updated.csv`, into a single dataframe. 
#
# * The merged dataframe contains:
#     - index: the time instance `timestamp` of the measurements
#     - columns: the location of the site `LocationName`, the sensor ID `SensorUnit_ID`, the CO2 measurement `CO2`, the `temperature`, the `humidity`, the `zone`, the `altitude`, the longitude `lon` and the latitude `lat`.
#
# | timestamp | LocationName | SensorUnit_ID | CO2 | temperature | humidity | zone | altitude | lon | lat |
# |:---------:|:------------:|:-------------:|:---:|:-----------:|:--------:|:----:|:--------:|:---:|:---:|
# |    ...    |      ...     |      ...      | ... |     ...     |    ...   |  ... |    ...   | ... | ... |
#
#
#
# * For each measurement (CO2, humidity, temperature), __take the average over an interval of 30 min__. 
#
# * If there are missing measurements, __interpolate them linearly__ from measurements that are close by in time.
#
# __Hints__: The following methods could be useful
#
# 1. ```python 
# pandas.DataFrame.resample()
# ``` 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
#     
# 2. ```python
# pandas.DataFrame.interpolate()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
#     
# 3. ```python
# pandas.DataFrame.mean()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html
#     
# 4. ```python
# pandas.DataFrame.append()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html

# %%
DATA_FOLDER = "../data/carbosense-raw/"

# %%
# read co2 measurements
co2_measurements_df = pd.read_csv(f'{DATA_FOLDER}CO2_sensor_measurements.csv', parse_dates=[0], sep='\t', on_bad_lines='warn')
display(co2_measurements_df.info())
co2_measurements_df.head()


# %%
# Get all the sensor ids
sensor_ids = co2_measurements_df.SensorUnit_ID.unique()
# Build dictionary of sensors ids to their locations, to reconstruct the dataframe later
sensors_locations_dict = dict(zip(co2_measurements_df.SensorUnit_ID, co2_measurements_df.LocationName))

# Pivot co2 measurements to get sensor id in each column (like temperature humidities dataframe below), to resample for each sensor
co2_measurements_df_resampled = co2_measurements_df.pivot(index='timestamp', columns='SensorUnit_ID', values='CO2')

# resample using 30 min interval and take the average of measurements in that interval
co2_measurements_df_resampled = co2_measurements_df_resampled.resample('30T').mean()

# interpolate linearly for Co2 measurements that are missing. Here we set limit direction to both to fill all NaN.
co2_measurements_df_resampled = co2_measurements_df_resampled.interpolate(method='linear', limit_direction='both')

# Return to original form using melt but now resampled
co2_measurements_df_resampled.reset_index(inplace=True)
co2_measurements_cleaned = co2_measurements_df_resampled.melt(
    id_vars='timestamp', value_vars=sensor_ids, var_name='SensorUnit_ID', value_name='CO2'
)
display(f'Number of measurements {len(co2_measurements_cleaned)}')
co2_measurements_cleaned.head()

# %%
# Finally fill back locations
co2_measurements_cleaned['LocationName'] = co2_measurements_cleaned.SensorUnit_ID.map(sensors_locations_dict)
co2_measurements_cleaned.head()

# %% [markdown]
# **Note** that we've resampled the CO2 measurements in periods of 30 minutes, **before** interpolating them linearly. If we were to interpolate before resampling, we could have a lot of missing values that will be interpolated, which might not reflect the actual values of CO2. After resampling, we'll have less NaNs to interpolate, which produces better results. This applies to the temperature and humidity measurements below as well.

# %%
# read temperature humidity measurements
temp_hum_df = pd.read_csv(f'{DATA_FOLDER}temperature_humidity.csv', parse_dates=[0], sep='\t', on_bad_lines='warn')
temp_hum_df.head()

# %%
# Resample using 30 min interval and take average
temp_hum_df_resampled = temp_hum_df.set_index('Timestamp').resample('30T').mean()

# interpolate linearly for temperature and humidity measurements that are missing. Here we set limit direction to both to fill all NaN.
temp_hum_df_resampled = temp_hum_df_resampled.interpolate(method='linear', limit_direction='both') 

temp_hum_df_resampled.reset_index(inplace=True)
temp_hum_df_resampled.head()

# %%
# create a dictionary to rename the columns
ren_cols = {x: x.split('.')[0] for x in temp_hum_df_resampled.columns}

# get the name of all the temperature columns
temp_cols =['Timestamp'] + [f'{x}.temperature' for x in sensor_ids]
# get the name of all the humifity columns
hum_cols = ['Timestamp'] + [f'{x}.humidity' for x in sensor_ids]

# Rename the columns in the temperature dataframe and put it in the correct format
temp_df = temp_hum_df_resampled[temp_cols].rename(columns=ren_cols)
temp_df = temp_df.melt(id_vars = 'Timestamp', value_vars = temp_df.columns, var_name = 'SensorUnit_ID', value_name='temperature')

# Rename the columns in the humidity dataframe and put it in the correct format
hum_df = temp_hum_df_resampled[hum_cols].rename(columns=ren_cols)
hum_df = hum_df.melt(id_vars = 'Timestamp', value_vars = hum_df.columns, var_name = 'SensorUnit_ID', value_name='humidity')

# merge the two dataframes together
temp_hum_cleaned = temp_df.merge(hum_df, on=['Timestamp', 'SensorUnit_ID'])
temp_hum_cleaned['SensorUnit_ID'] = temp_hum_cleaned['SensorUnit_ID'].astype(int)
temp_hum_cleaned

# %%
# Read sensors metadata measurements
sensors_metadata_df = pd.read_csv(f'{DATA_FOLDER}sensors_metadata_updated.csv', on_bad_lines='warn', index_col=0)
sensors_metadata_df.head()

# %%
# Merge everything into one single dataframe
sensors_metadata_df.rename(columns={'LAT': 'lat', 'LON': 'lon'}, inplace=True)
temp_hum_cleaned.rename(columns={'Timestamp':'timestamp'}, inplace=True)

co2_temp_hum = co2_measurements_cleaned.merge(
    temp_hum_cleaned, on=['timestamp', 'SensorUnit_ID']
)
measurements_cleaned = co2_temp_hum.merge(
    sensors_metadata_df[['LocationName', 'zone', 'altitude', 'lon', 'lat']], on='LocationName'
)

# %%
#Final dataframe
measurements_cleaned

# %% [markdown]
# ### b) **2/10** 
#
# Export the curated and ready to use timeseries to a csv file, and properly push the merged csv to Git LFS.

# %%
measurements_cleaned.to_csv(f'{DATA_FOLDER}measurements_cleaned.csv', index=False)

# %% [markdown]
# The following commands were used to add the file to git lfs.

# %%
# !git lfs track ../data/carbosense-raw/measurements_cleaned.csv

# %%
# !git add ../data/carbosense-raw/measurements_cleaned.csv

# %%
# !git commit -m "Cleaned measurements"
# !git push

# %%
# Check that it reads correctly
df = pd.read_csv(f'{DATA_FOLDER}measurements_cleaned.csv')

# %% [markdown]
# ## PART II: Data visualization (15 points)

# %% [markdown]
# ### a) **5/15** 
# Group the sites based on their altitude, by performing K-means clustering. 
# - Find the optimal number of clusters using the [Elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)). 
# - Wite out the formula of metric you use for Elbow curve. 
# - Perform clustering with the optimal number of clusters and add an additional column `altitude_cluster` to the dataframe of the previous question indicating the altitude cluster index. 
# - Report your findings.
#
# __Note__: [Yellowbrick](http://www.scikit-yb.org/) is a very nice Machine Learning Visualization extension to scikit-learn, which might be useful to you. 

# %%
# Retrieve the preprocessed measurements from part 1 (Don't forget to git lfs pull!)
measurements = pd.read_csv(f'{DATA_FOLDER}measurements_cleaned.csv', parse_dates=[0], on_bad_lines='warn') 
measurements.head()

# %%
# Get the sites and their altitudes
filter_cols = ['LocationName', 'altitude']
locations_altitudes = measurements[filter_cols].drop_duplicates(subset=filter_cols)
locations_altitudes = locations_altitudes.set_index('LocationName')
display(locations_altitudes.head())
locations_altitudes.info()

# %% [markdown]
# Here we use yellowbrick to the see a vizualization of the elbow to find the optimal number of clusters for the sites altitudes. There are different metrics to calculate the elbow score. We are going to use the default one: the **distortion** which computes the sum of squared distances from each point to its assigned center: 
#
# $$distortion = \Sigma_n ||x_n - c(x)||^2$$
#
# where $c(x)$ is the centroid of the cluster to which $x_n$ is assigned to. We are going to try and find the optimal number of clusters ranging from **2 to 8 clusters** using the elbow method.
#
# We are also going to use the **silhouette coefficient** metric, so we can compare the results between the 2 metrics.

# %%
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Instantiate the clustering model kmeans and the yellow brick elbow visualizer using distortion metric
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,8), timings=False)

# Fit the data to the visualizer
visualizer.fit(locations_altitudes)   
# Finalize and render the figure
visualizer.show()
plt.show()

# %% [markdown]
# - With distortion the **optimal number of clusters is 4**. Note that increasing the minimum number of clusters fed to the elbow method might increase the optimal as well. But since we're using the number of clusters to do analysis on CO2 as a function of the height, it's maybe best to keep it low.

# %% [markdown]
# Now we are going to compare this with the results of running the elbow method using the [silhouette coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering)), which is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).

# %%
# Instantiate the clustering model kmeans and the yellow brick elbow visualizer using the silhouette metric
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,8), timings=False, metric='silhouette')

# Fit the data to the visualizer
visualizer.fit(locations_altitudes)
# Finalize and render the figure
visualizer.show()
plt.show()

# %% [markdown]
# - As we can see, the silhouette score gives that **2 clusters suffice** to cluster the sites by their altitudes.

# %% [markdown]
# Now we run Kmeans with k as the optimal number of clusters found above.

# %%
from sklearn.metrics import silhouette_score

def calculate_kmeans(test_df, k=4):
    kmeans = KMeans(init = 'random', n_clusters=k, random_state=42)
    kmeans.fit(locations_altitudes)
    print(f'K={k}, Altitude cluster centers:{kmeans.cluster_centers_} with silhouette score {silhouette_score(test_df, kmeans.labels_)}')
    return kmeans.cluster_centers_, kmeans.labels_
          
cluster_centers_4, labels_4 = calculate_kmeans(locations_altitudes, 4)
cluster_centers_3, labels_3 = calculate_kmeans(locations_altitudes, 3)
cluster_centers_2, labels_2 = calculate_kmeans(locations_altitudes, 2)

# %% [markdown]
# - We're going to cluster the sites by their altitudes using the optimal number of clusters **k = 4** found by the distortion metric, because the 2 clusters optimal number found by the silhouette metric might not be very informative when analyzing CO2 as a function of altitude. Also note that with k=3 or k = 4, there is one location **UTLI** (863.6m) the alitude of which is so high, it's going to be grouped into its own cluster.

# %%
# Append the cluster number to the measurements df
locations_clusters_dict = dict(zip(locations_altitudes.index, labels_4))
measurements['altitude_cluster_4'] = measurements['LocationName'].map(locations_clusters_dict)

# Cluster index with 2 clusters
locations_clusters_dict_2 = dict(zip(locations_altitudes.index, labels_2))
measurements['altitude_cluster_2'] = measurements['LocationName'].map(locations_clusters_dict_2)

measurements.head()

# %%
# Visualize the distributions of altitudes within each cluster
fig = px.box(measurements, y='altitude', x='altitude_cluster_4')
fig.update_layout(
                yaxis_title='Altitude [m]',
                xaxis_title='Clusters',
                title='Distributions of altitudes within each cluster (4 clusters)',
                hovermode="x")

# %%
# Visualize the distributions of altitudes within each cluster
fig = px.box(measurements, y='altitude', x='altitude_cluster_2')
fig.update_layout(
                yaxis_title='Altitude [m]',
                xaxis_title='Clusters',
                title='Distributions of altitudes within each cluster (2 clusters)',
                hovermode="x")

# %% [markdown]
# From the above boxplots, we can notice that the clustering for $k=2$ leads to more outliers compared to clustering with $k=4$. We will therefore keep $k=4$ for the next steps.

# %%
measurements.drop(columns=['altitude_cluster_2'], inplace=True, errors='ignore')
measurements.rename(columns={'altitude_cluster_4': 'altitude_cluster'}, inplace=True)

# %% [markdown]
# ### b) **4/15** 
#
# Use `plotly` (or other similar graphing libraries) to create an interactive plot of the monthly median CO2 measurement for each site with respect to the altitude. 
#
# Add proper title and necessary hover information to each point, and give the same color to stations that belong to the same altitude cluster.

# %%
# Calculate October CO2 median measurement for each site (since each site has its own unique altitude)
co2_medians_by_site = measurements.groupby(by=['LocationName', 'altitude']).agg(median_CO2=('CO2', 'median'))
co2_medians_by_site = co2_medians_by_site.reset_index()
co2_medians_by_site['altitude_cluster'] = co2_medians_by_site['LocationName'].map(locations_clusters_dict)
co2_medians_by_site = co2_medians_by_site.sort_values('median_CO2')
co2_medians_by_site.head()

# %%
# Convert altitude cluster to string for plotly to interpret it as categorical variables.
co2_medians_by_site['altitude_cluster'] = co2_medians_by_site['altitude_cluster'].astype(str)
fig = px.scatter(co2_medians_by_site, x='altitude', y='median_CO2', color='altitude_cluster', 
                 labels={'altitude':'Altitude','median_CO2':'CO2', "altitude_cluster":"Cluster", "LocationName":"Name"}, 
                 hover_data=['LocationName'])
fig.update_layout(
                yaxis_title='Median CO2 over the month',
                xaxis_title='Altitude',
                title='Clustering based on the altitude',
                hovermode="x")

# %% [markdown]
# ### c) **6/15**
#
# Use `plotly` (or other similar graphing libraries) to plot an interactive time-varying density heatmap of the mean daily CO2 concentration for all the stations. Add proper title and necessary hover information.
#
# __Hints:__ Check following pages for more instructions:
# - [Animations](https://plotly.com/python/animations/)
# - [Density Heatmaps](https://plotly.com/python/mapbox-density-heatmaps/)

# %%
mean_by_day = measurements.copy()

mean_by_day['day'] = mean_by_day.timestamp.map(lambda t: t.day)
mean_by_day = mean_by_day.groupby(['LocationName','day']).mean().reset_index()
mean_by_day.head()

# %%
import math

# some values for a prettier plot
min_co2 = math.floor(mean_by_day.CO2.min())
max_co2 = math.ceil(mean_by_day.CO2.max())
center = dict(lat=mean_by_day.lat.mean(), lon=mean_by_day.lon.mean())

# density heatmap of the mean daily CO2 concentration
fig = px.density_mapbox(mean_by_day, lat='lat', lon='lon', z='CO2', radius=20,
                        opacity=0.75,
                        center=center, zoom=10,
                        mapbox_style='open-street-map', 
                        animation_frame='day', 
                        hover_name='LocationName', 
                        hover_data={'humidity':':.2f', 'temperature':':.2f', 'altitude':':.2f'}, 
                        range_color=[min_co2, max_co2],
                        color_continuous_scale=[(0, px.colors.sequential.Plasma[0]),
                                                (0.01, px.colors.sequential.Plasma[2]),
                                                (0.05, px.colors.sequential.Plasma[4]),
                                                (0.12, px.colors.sequential.Plasma[5]),
                                                (0.2, px.colors.sequential.Plasma[6]),
                                                (0.5, px.colors.sequential.Plasma[8]),
                                                (1, px.colors.sequential.Plasma[9])],
                        width=800,
                        height=800
                       )
fig.update_layout(title = {
    'text': 'Mean daily CO2 concentration - October 2017'
})
fig.show()

# %% [markdown]
# ## PART III: Model fitting for data curation (35 points)

# %% [markdown]
# ### a) **2/35**
#
# The domain experts in charge of these sensors report that one of the CO2 sensors `ZSBN` is exhibiting a drift on Oct. 24. Verify the drift by visualizing the CO2 concentration of the drifting sensor and compare it with some other sensors from the network. 

# %%
# Load the measurements dataframe
measurements = pd.read_csv(f'{DATA_FOLDER}measurements_cleaned.csv')

# %%
# Make sure that there is no measurement with nan
measurements[measurements.isna()['temperature']]

# %%
from ipywidgets import interactive, widgets, interact
def plot_co2(locations=[]):
    """
    Plot CO2 curve for the current 
    """
    if not locations:
        return
    
    fig = px.line(measurements[measurements['LocationName'].isin(list(locations)+['ZSBN'])],x='timestamp',y='CO2',color='LocationName',labels={'timestamp':'Date','CO2':'CO2 (ppm)',"LocatioName":"Name"}, title='CO2 level October 24, 2017')
    fig.update_xaxes(dtick=24*60*60*1000, ticklabelmode='period')
    fig.update_layout(hovermode="x unified", yaxis_type="log",yaxis_title='CO2 (ppm) (log)')
    fig.show()
    
others = sorted(measurements.LocationName.unique())
others.remove('ZSBN')

location_selector = widgets.SelectMultiple(
    options=others,
    value= ['BSCR'],
    description='Sensors',
    disabled=False
)

# Insert code here
_ = interact(plot_co2,  locations = location_selector)

# %% [markdown]
# We can see that the measurements made by sensor ZSBN start to drift from October 24. Indeed, we can see a kind of periodic pattern during the first part of the month (with similar values) and since October 24, we have a clear "fall" in the CO2 measurement and we can see that we lost ~80 of the CO2 measurements. Furthermore, we can see that if we select any other sensor, the measurement continue to say "at the same level" for the end of the october month. 

# %% [markdown]
# ### b) **8/35**
#
# The domain experts ask you if you could reconstruct the CO2 concentration of the drifting sensor had the drift not happened. You decide to:
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features the covariates not affected by the malfunction (such as temperature and humidity)
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __confidence interval__ obtained from cross validation
# - What do you observe? Report your findings.
#
# __Note:__ Cross validation on time series is different from that on other kinds of datasets. The following diagram illustrates the series of training sets (in orange) and validation sets (in blue). For more on time series cross validation, there are a lot of interesting articles available online. scikit-learn provides a nice method [`sklearn.model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
#
# ![ts_cv](https://player.slideplayer.com/86/14062041/slides/slide_28.jpg)

# %% [markdown]
# ------
#  - We use $k=23$ folds for the cross validation, in other words, a fold per day.
#  - We construct the 95% confidence interval as follows :
#      - Compute the Root Mean Squared Error (RMSE, to have the same units as the prediction) on each fold, obtained via the Time Series Cross Validation (TSCV)
#      - Compute a 95% confidence interval (CI) for the average RMSE via bootstrap, resampling 9999 times (default value in scipy.stats)
#      - Compute the average RMSE over these 23 folds
#      - Compute the difference between the upper bound of the CI and the average RMSE (1) as well the one between the average RMSE and the lower bound of the CI (2)
#      - Plot the predictions for the entire month of the regression trained on all the data pre-drift
#      - Plot the CI around these predictions taking the upper bound as being the prediction + (1) and the lower bound being the prediction - (2). In our opinion this is not the most accurate way to characterize the accuracy of the predictions. But we think it is the most "correct" way to do it compared to what was asked and said in the slack forum. In particular, this assumes that the error is stationary as we use the same CI for all the points. We could also not use the Gaussian trick of taking the average +/- 2* standard deviation as it was specified we need to build an "exact" 95% CI and not a 95.45%.

# %%
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import bootstrap

def fit_TSCV(X_before, y_before, X_month, cols_to_take):
    """
    Perform the time series cross validation (TSCV) and output the mean and standard deviation of the root mean squared error. 
    Parameters:
    ------------
        - X_before: the dataset containing all the points for training
        - y_before: the expected output of the training set
        - X_month: The measurements for the whole month
        - cols_to_take: the list of columns to consider as features.
        
    Return:
    ------------
        - mean of the RMSE errors
        - 95% CI for the RMSE (bootstraped)
    """
    # Get only the requested column
    X_before = X_before.loc[:, cols_to_take]
    X_month = X_month.loc[:, cols_to_take]
    
    # Timeseries cross validation
    reg = LinearRegression(n_jobs=-1)
    splitter = TimeSeriesSplit(n_splits=23)
    
    errors = []

    for i, (train_index, test_index) in enumerate(splitter.split(X_before)):
        # Select the train and test dataset
        X_train, y_train = X_before.iloc[train_index], y_before.iloc[train_index]
        X_test, y_test = X_before.iloc[test_index], y_before.iloc[test_index]
        # Train and compute the errors 

        lin_reg = reg.fit(X_train, y_train)
        y_test_pred = lin_reg.predict(X_test)
        
        # save the rmse 
        errors.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        
    # Compute avg and 95% CI via bootstrap
    return np.mean(errors), bootstrap((errors,), np.mean)


# %%
# Split the dataset and take only the relevant columns
X_before = measurements.loc[(measurements['LocationName'] == 'ZSBN') & (measurements['timestamp'] < '2017-10-24'), ['temperature', 'humidity','timestamp']]
y_before = measurements.loc[(measurements['LocationName'] == 'ZSBN') & (measurements['timestamp'] < '2017-10-24'), 'CO2']
X_month = measurements.loc[(measurements['LocationName'] == 'ZSBN'), ['temperature', 'humidity', 'timestamp']]
y_month = measurements.loc[(measurements['LocationName'] == 'ZSBN'), 'CO2']

# Compute the mean and standard deviation of the TSCV
mean, ci = fit_TSCV(X_before, y_before, X_month, ['temperature', 'humidity'])


# %%
def fit_predict_display(X_before, y_before, X_month, y_month, cis, rmse_mean, cols_to_take):
    """
    Fit the model to the training data and predict for the whole month
    Parameters:
    ------------
        - X_before: The training data
        - y_before: The outcome of training data
        - X_month: The feature for the whole month
        - y_month: the outcome expected for the whole month
        - cis: the boostraped ci obtained during cross-validation
        - rmse_mean: mean RMSE on the cross validation
        - cols_to_take: The columns to take into account for training and prediction
    """
    # Select only the relevant columns for training and prediction
    X_before_feat = X_before.loc[:,cols_to_take]
    X_month_feat = X_month.loc[:,cols_to_take]
    
    # Train on the data points before drift
    lin_reg = LinearRegression(n_jobs=-1)
    lin_reg = lin_reg.fit(X_before_feat, y_before)

    # Predict data for entire month
    y_month_pred = lin_reg.predict(X_month_feat)

    # Convert back to dataframe
    df_preds = pd.DataFrame()
    df_preds['timestamp'] = X_month['timestamp']
    df_preds['CO2_pred'] = y_month_pred
    df_preds['CO2'] = y_month.values
    
    # Build CI around prediction as explained above
    df_preds['CI_high'] = df_preds['CO2_pred'] + (cis.confidence_interval.high - rmse_mean)
    df_preds['CI_low'] = df_preds['CO2_pred'] - (rmse_mean - cis.confidence_interval.low)
    
    # Plot the predictions and actual data
    plot_predictions(df_preds)


# %%
def plot_predictions(df_preds):
    """
    Plot the prediction lines with confidence interval and the actual CO2 measurement
    Parameters: 
    ------------
        - df_preds: The dataframes containing actual measurement and predictions
    """
    fig = go.Figure()
    fig.add_traces([
        # The confidence interval curves
        go.Scatter(
            x = df_preds['timestamp'], 
            y = df_preds['CI_low'],
            mode = 'lines', 
            name="Lower 95% CI",
            line_color = 'rgba(0,0,255,0)',
            showlegend = False),
        go.Scatter(x = df_preds['timestamp'], y = df_preds['CI_high'],
            mode = 'lines', line_color = 'rgba(0,0,0,0)',
            name = 'Upper 95% CI',
            fill='tonexty', fillcolor = 'rgba(0, 0, 255, 0.2)'),
        # Actual measurement
        go.Scatter(name='Actual CO2 measurements',
            x=df_preds['timestamp'],
            y=df_preds['CO2'],
            mode='lines',
            line=dict(color='red')
        ),
        # Prediction line
        go.Scatter(name='Predicted CO2 measurements',
            x=df_preds['timestamp'],
            y=df_preds['CO2_pred'],
            mode='lines',
            line=dict(color='blue')
        )
    ])
    # Update global info of the graph
    fig.update_layout(
        yaxis_title='CO2 (ppm)',
        xaxis_title='October 2017',
        title='CO2 measurements predictions',
        hovermode="x"
    )
    
    fig.show()


# %%
fit_predict_display(X_before, y_before, X_month, y_month, ci, mean, ['humidity', 'temperature'])

# %%
print(f'95% Confidence Interval for Root Mean Square Error : [{ci.confidence_interval.low:2f}; {ci.confidence_interval.high:2f}]')
print(f'Average Root Mean Square Error : {mean:2f}')


# %% [markdown]
# First, we can see that the linear regression fits the seasonality of the CO2 measurements.
#
# By training the linear regression on data before the drift and predicting the data points of the entire month with this model, we are able to confirm that the data after the 24 october do not follow the same pattern as before. Indeed, we clearly see that the predicted data continue to stay around the same level whereas the actual measurements suddenly drop around the 24 october. For the above plot, the confidence interval is computed as explained at the beginning of part b) above. You might want to zoom in a bit to see it more clearly (light blue).

# %% [markdown]
# ### c) **10/35**
#
# In your next attempt to solve the problem, you decide to exploit the fact that the CO2 concentrations, as measured by the sensors __experiencing similar conditions__, are expected to be similar.
#
# - Find the sensors sharing similar conditions with `ZSBN`. Explain your definition of "similar condition".
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features:
#     - the information of provided by similar sensors
#     - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __confidence interval__ obtained from cross validation
# - What do you observe? Report your findings.

# %% [markdown]
# -----
# To determine what are sensors in similar conditions, we choose to base this on the 3 measures: humidity, temperature and altitude. To determine the closest neighbour, we proceed as follows :
#  - Compute the average temperature, humidity for each sensor
#  - Standardize these data so that all three measures will have the same weights in the decision as they are not in the same units
#  - Finally, we compute the $L2$ norm between the measures of avg humidity, avg temperature and altitude of the ZSBN sensor compared to the measures of all the other sensors
#  - Consider only the k nearest in terms of this $L2$ distance

# %%
def get_similar_conditions_sensors(measurements, metadata, k=5, features=['temperature']):
    """
    Get the k nearest sensors according to their altitude
    Parameters: 
    ------------
        - measurements: dataframe containing all measurements
        - metadata: dataframe containing information about the sensors
        - k: number of sensors to keep (excluding ZSBN)
        - features: features to consider as "similar" conditions
    
    Return:
    ------------
        - dataframe containing data of the k nearest sensors
    """
    # Compute mean temperature and humidity by sensor
    avgs = measurements.groupby('LocationName').mean()[['temperature', 'humidity']].reset_index()
    
    # Merge to get altitude data
    df = avgs.merge(metadata[['LocationName', 'altitude']], how='inner', on='LocationName')
    
    # Standardize the features so that they have same weight
    df[features] = (df[features] - df[features].mean()) / df[features].std()
    
    # Get avg data for ZSBN sensor
    zsbn = df[df['LocationName'] == 'ZSBN']
    
    # Compute difference in L2 norm
    df['diff'] = df.apply(lambda row: np.linalg.norm(row[features] - zsbn[features]), axis=1)
    
    # Keep only k nearest
    return df[df['LocationName'] != 'ZSBN'].nsmallest(k, 'diff')

def CV_k_nearest(measurements, metadata, k, features):
    """
    Select the k nearest (in terms of altitude) sensors and
    perform a cross-validation.
    Parameters: 
    ------------
        - measurements: dataframe containing all measurements
        - metadata: dataframe containing information about the sensors
        - k: number of sensors to keep (excluding ZSBN)
        - features: features to consider as "similar" conditions
        
    Return:
    ------------
        - mean_rmse : mean RMSE during CV
        - ci : 95% CI of the RMSE
        - X_before : features before drift
        - y_before : target before drift
        - X_month : features on entire month
        - y_month : target on entire month
    """
    # Get k nearest sensors
    k_nearest = get_similar_conditions_sensors(measurements, metadata, k, features)
    df_nearest = measurements[measurements['LocationName'].isin(k_nearest['LocationName'].tolist() + ['ZSBN'])]
    
    # Select data from these k sensors + ZSBN
    data = df_nearest[['timestamp', 'LocationName', 'temperature', 'humidity', 'CO2']].pivot(index='timestamp', columns='LocationName', values=['temperature', 'humidity', 'CO2'])
    data.columns = ['_'.join(col) for col in data.columns.values]
    
    cols_zsbn_test = [i for i in data.columns if 'ZSBN' in i and 'CO2' in i]
    cols_train= [i for i in data.columns if ('CO2' not in i) or ('ZSBN' not in i and 'CO2' in i)]
    
    # Select data before the drift
    X_before = data.iloc[data.index < '2017-10-24',:].loc[:, cols_train].reset_index()
    y_before = data.iloc[data.index < '2017-10-24',:].loc[:, cols_zsbn_test]
    X_month = data.loc[:, cols_train].reset_index()
    y_month = data.loc[:, cols_zsbn_test]
    
    # Perform cross validation to compute the RMSE
    mean_rmse, ci = fit_TSCV(X_before, y_before, X_month, X_before.columns[1:])
    
    return mean_rmse, ci, X_before, y_before, X_month, y_month

def compute_error(measurements, metadata, features, limit=23):
    """
    Compute the RMSE for different values
    of k (number of sensors to keep).
    Parameters: 
    ------------
        - measurements: dataframe containing all measurements
        - metadata: dataframe containing information about the sensors
        - limit: max number of sensors to keep (excluding ZSBN)
        - features: features to consider as "similar" conditions
        
    Return:
    ------------
        - means : mean RMSE during CV for all k's
    """
    means = []
    for k in range(1,1+limit):
        # Compute CV rmse
        ret = CV_k_nearest(measurements, metadata, k, features)
        means.append(ret[0])
        
    return means
        
def plot_errors_vs_k(means):
    """
    Plot the RMSE for different values
    of k (number of sensors to keep).
    Parameters: 
    ------------
        - means: list containing the mean RMSE for the k's
    """
    # Plot errors for different k's
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,1+len(means))), y=means, mode='lines+markers'))
    fig.update_layout(
        yaxis_title='Average Root mean squared error',
        xaxis_title='k',
        title='Average root mean squared error using k',
        hovermode="x"
    )
    
    fig.show()


# %%
# Load information about sensors
sensors_metadata_df = pd.read_csv(f'{DATA_FOLDER}sensors_metadata_updated.csv', on_bad_lines='warn', index_col=0)

# %% [markdown]
# Below we try every possible combinations of features to determine which one gives the lowest mean RMSE on the test set. By doing so, we observe that taking all the three measures to define the "closest neighbours" doesn't gave the best results.

# %%
from itertools import combinations

def powerset(iterable):
    for n in range(1, len(iterable) + 1):
        yield from combinations(iterable, n)

# Try all combinations of temperature, humidity, altitude
for cols in powerset(['altitude', 'temperature', 'humidity']):
    # Compute RMSE for all the k's and keep the best
    cols = list(cols)
    print(cols)
    best_k = np.argmin(compute_error(measurements, sensors_metadata_df, cols, limit=20)) + 1
    print(f'Best k for columns {cols[0]} : {best_k}')
    
    # Perform cross-validation
    mean, ci, X_before, y_before, X_month, y_month =  CV_k_nearest(measurements, sensors_metadata_df, k=best_k, features=cols)
    
    print(f'95% Confidence Interval for Root Mean Square Error : [{ci.confidence_interval.low:2f}; {ci.confidence_interval.high:2f}]')
    print(f'Average Root Mean Square Error : {mean:2f}')
    print('-------------------------------------------------------------------------------')

# %% [markdown]
# From the above computation, we can see that the best results are obtained by considering only the temperature as a criterion of similarity between two sensors as they obtain the lowest average RMSE as well as the smallest confidence interval (obtained via boostrapping) during cross validation.
#
# However, we can see that the difference between the RMSE with the temperature only and the altitude only is not significant. As we used the altitude data in part 2 to cluster the sensors, we will only consider the altitude as "similar conditions".

# %%
# Cross validate the different values of k (number of sensors to keep)
features = ['altitude']
means = compute_error(measurements, sensors_metadata_df, features=features, limit=20)
plot_errors_vs_k(means)

# %% [markdown]
# The above plot shows the different values of the avg RMSE using different numbers of similar sensors. We can observe that considering 4 sensors seems to give the best RMSE. This will yield 4*3 + 2 = 14 features.

# %%
get_similar_conditions_sensors(measurements, sensors_metadata_df, k=4, features=['temperature'])

# %%
# Perform cross validation considering as features the data of the 4 nearest sensors
mean, ci, X_before, y_before, X_month, y_month =  CV_k_nearest(measurements, sensors_metadata_df, k=4, features=features)

# Fit a linear regression on data pre-drift and predict entire month
fit_predict_display(X_before, y_before, X_month, y_month, ci, mean, X_before.columns[1:])

# %%
print(f'95% Confidence Interval for Root Mean Square Error : [{ci.confidence_interval.low:2f}; {ci.confidence_interval.high:2f}]')
print(f'Average Root Mean Square Error : {mean:2f}')


# %% [markdown]
# The avg RMSE obtained via CV is much smaller than the one from part b). This means that considering data from similar sensors helps a lot in predicting the CO2 level of the ZSBN sensor.
#
# Similarly to part b), we have this clear drift from the 24th of October. The model fits fairly well the actual measurements at first sight. However, from the 24th, the actual measurements are dropping whereas the model, trained on data before this drift, continues to predict values at levels similar to the period before the 24th. The predictions after the drift seem to fit fairly well the shape of the actual data. So this suggests the actual data are just shifted by a constant to a lower value.
#
# You might want to zoom in a bit to see it more clearly (light blue).

# %% [markdown]
# ### d) **10/35**
#
# Now, instead of feeding the model with all features, you want to do something smarter by using linear regression with fewer features.
#
# - Start with the same sensors and features as in question c)
# - Leverage at least two different feature selection methods
# - Create similar interactive plot as in question c)
# - Describe the methods you choose and report your findings

# %% [markdown]
# ----
# #### Method 1 : Aggregation of features
# A very simple method to reduce the number of feature and reduce noise in the data is to average the different quantities over the k neighboring sensors. We also consider the standard deviation of these measurements. So we consider the following 6 features (instead of 14 before) :
# - Temperature of ZSBN
# - Humidity of ZSBN
# - Avg temperature of the 4 nearest sensors (from c) : ZORL, ZDLT, ZWCH, ZECB
# - Avg humidty of the 4 nearest sensors (from c) : ZORL, ZDLT, ZWCH, ZECB
# - Avg CO2 of the 4 nearest sensors (from c) : ZORL, ZDLT, ZWCH, ZECB
# - Std CO2 of the 4 nearest sensors (from c) : ZORL, ZDLT, ZWCH, ZECB

# %%
def aggregate_features(df):
    """
    Aggregate the features of different sensors by taking avg and std
    for each measurement
    Parameters: 
    ------------
        - df: dataframe containing sensors measurements
        
    Return:
    ------------
        - res : dataframe with aggregated features
    """
    # Keep temperature and humidity of ZSBN untouched
    res = df[['timestamp', 'temperature_ZSBN', 'humidity_ZSBN']]
    
    # Take avg and std of temperature, humidity and CO2 on other sensors
    cols_temp = [i for i in df.columns if 'temperature' in i and 'ZSBN' not in i]
    cols_hum = [i for i in df.columns if 'humidity' in i and 'ZSBN' not in i]
    res['temperature_avg'] = df[cols_temp].mean(axis=1)
    res['humidity_avg'] = df[cols_hum].mean(axis=1)
    
    cols_CO2 = [i for i in df.columns if 'CO2' in i]
    res['CO2_avg'] = df[cols_CO2].mean(axis=1)
    res['CO2_std'] = df[cols_CO2].std(axis=1)
    
    return res


# %%
# Aggregate the features
X_month_agg = aggregate_features(X_month)
X_before_agg = X_month_agg[X_month_agg['timestamp'] < '2017-10-24']

# %%
# Perform CV
mean_agg, ci_agg = fit_TSCV(X_before_agg, y_before, X_month_agg, X_month_agg.columns != 'timestamp')

# Fit model on data pre-drift and predict entire month
fit_predict_display(X_before_agg, y_before, X_month_agg, y_month, ci_agg, mean_agg, X_before_agg.columns[1:])

# %%
print(f'95% Confidence Interval for Root Mean Square Error : [{ci_agg.confidence_interval.low:2f}; {ci_agg.confidence_interval.high:2f}]')
print(f'Average Root Mean Square Error : {mean_agg:2f}')

# %% [markdown]
# Here we can see that the confidence interval for the RMSE is a bit smaller than with all the features from part c. The mean RMSE is however very similar to part c. This suggests that knowing only the aggregated information from other sensors is enough to perform as well as the full model. It is actually slightly more performant. We also still clearly see the drift beginning on the 24th of October.

# %% [markdown]
# #### Method 2 : Selection of features
#
# A second method is to select specific features that convey the most information about the target. There are multiple ways to do this. For example, we could consider as score metric the mutual information between the covariate and the target. However, the mutual information captures any relationship between the variables. Here we privileged using the F-statistic which can be used to determine if a nested model fit significantly better the data than its unrestricted version.
#
# We decided to keep also 6 features to be consistent with method 1 regarding the dimensionality of the features.

# %%
from sklearn.feature_selection import f_regression, SelectKBest

def select_features(df, y, k=6):
    """
    Select the k best features according to their
    F-statistic.
    Parameters: 
    ------------
        - df: dataframe containing sensors measurements
        - y: target
        - k: number of features to keep
        
    Return:
    ------------
        - res : dataframe with selected features
    """
    # Use F-statistic to select the features
    selector = SelectKBest(f_regression, k=k).fit(df.loc[:, df.columns != 'timestamp'], y['CO2_ZSBN'])
    cols_name = selector.get_feature_names_out()
    
    return df.loc[:, ['timestamp'] + list(cols_name)]


# %%
# Select the k best features
X_month_sel = select_features(X_month, y_month, k=6)

print(f'Selected features : {X_month_sel.columns.tolist()}')

X_before_sel = X_month_sel.loc[X_month_sel['timestamp'] < '2017-10-24', :]

# %%
# Perform CV
mean_sel, ci_sel = fit_TSCV(X_before_sel, y_before, X_month_sel, X_month_sel.columns != 'timestamp')

# Fit model on data pre-drift and predict entire month
fit_predict_display(X_before_sel, y_before, X_month_sel, y_month, ci_sel, mean_sel, X_before_sel.columns[1:])

# %%
print(f'95% Confidence Interval for Root Mean Square Error : [{ci_sel.confidence_interval.low:2f}; {ci_sel.confidence_interval.high:2f}]')
print(f'Average Root Mean Square Error : {mean_sel:2f}')

# %% [markdown]
# Here we observe that the avg RMSE is slightly lower than that of method 1. We again have this clear drift from the 24th of October.

# %% [markdown]
# ### e) **5/35**
#
# Eventually, you'd like to try something new - __Bayesian Structural Time Series Modelling__ - to reconstruct counterfactual values, that is, what the CO2 measurements of the faulty sensor should have been, had the malfunction not happened on October 24. You will use:
# - the information of provided by similar sensors - the ones you identified in question c)
# - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
#
# To answer this question, you can choose between a Python port of the CausalImpact package (such as https://github.com/jamalsenouci/causalimpact) or the original R version (https://google.github.io/CausalImpact/CausalImpact.html) that you can run in your notebook via an R kernel (https://github.com/IRkernel/IRkernel).
#
# Before you start, watch first the [presentation](https://www.youtube.com/watch?v=GTgZfCltMm8) given by Kay Brodersen (one of the creators of the causal impact implementation in R), and this introductory [ipython notebook](https://github.com/jamalsenouci/causalimpact/blob/HEAD/GettingStarted.ipynb) with examples of how to use the python package.
#
# - Report your findings:
#     - Is the counterfactual reconstruction of CO2 measurements significantly different from the observed measurements?
#     - Can you try to explain the results?

# %%
from causalimpact import CausalImpact

# %%
# Get features of the 4 closest sensor (as in part c)
_, _, X_before, y_before, X_month, y_month =  CV_k_nearest(measurements, sensors_metadata_df, k=4, features=features)

# %%
# Create the needed datafra,e
full_data = X_month.copy()
y_month.rename(columns={'CO2_ZSBN':"y"},inplace=True)
#causal impact take the first column as the value to predict so we should put CO2_ZSBN as the first column
full_data = y_month.merge(full_data, right_on = 'timestamp', left_index=True)

# %%
full_data.drop('timestamp',axis=1)
pre = [0, len(full_data[full_data['timestamp']<'2017-10-24'])-1]
post = [pre[1]+1, len(full_data)-1]

# %%
# Fit the Causal impact and display the results. 
impact = CausalImpact(full_data.drop('timestamp',axis=1),pre, post, model_args={"niter":5000})
impact.run()

# %%
impact.plot()

# %%
impact.summary("report")

# %% [markdown]
# The first important thing to see is that the causal impact model tellus that indeed something happen. Indeed, we can see that the model (which is also based on the measurements from the other sensors) predict that if no action was taken (i.e. if the sensor didn't have a drift), the predicted measurements will keep their periodical oscillation (dashed line). 
#
# We can see both from the text that the drift of the sensor is statistically significant.

# %% [markdown]
# # That's all, folks!
