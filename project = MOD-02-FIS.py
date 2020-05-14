#!/usr/bin/env python
# coding: utf-8

# # Introduction: Pricing of Homes in King County, WA

# ![](house.png)

# 
# Welcome to my kernel
# 
# In this dataset we have to predict the sales price of houses in King County, Seattle. It includes homes sold between May 2014 and May 2015. Before doing anything we should first know about the dataset what it contains what are its features and what is the structure of data.
# 
# The dataset cantains 21 house features plus the price, along with 21597 observations.
# 
# The description for the 21 features is given below:
# 
# 1. id :- It is the unique numeric number assigned to each house being sold.
# 2. date :- It is the date on which the house was sold out.
# 3. price:- It is the price of house which we have to predict so this is our target variable and aprat from it are our features.
# 4. bedrooms :- It determines number of bedrooms in a house.
# 5. bathrooms :- It determines number of bathrooms in a bedroom of a house.
# 6. sqft_living :- It is the measurement variable which determines the measurement of house in square foot.
# 7. sqft_lot : It is also the measurement variable which determines square foot of the lot.
# 8. floors: It determines total floors means levels of house.
# 9. waterfront : This feature determines whether a house has a view to waterfront 0 means no 1 means yes.
# 10. view : This feature determines whether a house has been viewed or not 0 means no 1 means yes.
# 11. condition : It determines the overall condition of a house on a scale of 1 to 5.
# 12. grade : It determines the overall grade given to the housing unit, based on King County grading system on a scale of 1 to 11.
# 13. sqft_above : It determines square footage of house apart from basement.
# 14. sqft_basement : It determines square footage of the basement of the house.
# 15. yr_built : It detrmines the date of building of the house.
# 16. yr_renovated : It detrmines year of renovation of house.
# 17. zipcode : It determines the zipcode of the location of the house.
# 18. lat : It determines the latitude of the location of the house.
# 19. long : It determines the longitude of the location of the house.
# 20. sqft_living15 : Living room area in 2015(implies-- some renovations)
# 21. sqft_lot15 : lotSize area in 2015(implies-- some renovations)
# 
# Now, we know about the overall structure of a dataset . So let's apply some of the steps that we should generally do while applying OLS stats model.
# 

# # STEP 1: IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import style
import matplotlib.cm as cm
from matplotlib import *
from scipy.stats import pearsonr
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Exploring the whole dataset
# 
# To get a sense for what is in the King County Housing dataset, first I will do some basic exploration of the entire dataset. After cleaning, another round of exploration will help clarify trends in the data specific to midrange housing.

# In[2]:


# Import needed packages and read in the data

data = pd.read_csv('kc_house_data.csv')


# In[3]:


# View first few rows of the dataset
data.head()


# # STEP 2: DATA CLEANING AND PREPROCESSING
# 
# In this step we check whether data contain null or missing values. What is the size of the data. What is the datatype of each column. What are unique values of categorical variables etc.

# In[4]:


# View counts and data types by column
data.info()


# In[5]:


# Check for missing values by column
data.isna().sum()


# In[6]:


# Check for duplicate records
print('Number of duplicate records: ', sum(data.duplicated()))


# In[7]:


# Check for duplicate IDs
display(data['id'].value_counts().head())

# Count non-unique IDs
id_value_counts = data['id'].value_counts()
num_repeat_ids = len(id_value_counts[id_value_counts > 1])*2 + 1
print('Number of non-unique IDs: ', num_repeat_ids)


# In[8]:


# Inspect a few of the records with duplicate IDs
display(data[data['id'] == 795000620])
display(data[data['id'] == 1825069031])
display(data[data['id'] == 2019200220])
display(data[data['id'] == 7129304540])
display(data[data['id'] == 1781500435])


# In[9]:


data.describe()


# In[10]:


# Check the number of unique values in each column
unique_vals_list = []
for col in data.columns:
    unique_vals_list.append({'column': col, 'unique values': len(data[col].unique())})
pd.DataFrame(unique_vals_list)


# In[11]:


# Define a function to create histograms
def hist_it(data):
    
    """Creates histograms of all numeric columns in a DataFrame"""
    
    data.hist(figsize=(16,14))


# In[12]:


# Create histograms for numerical variables
data_for_hist = data.drop(['id'], axis=1)

hist_it(data_for_hist)


# # STEP 3 : FINDING CORRELATION
# 
# 
# In this step we check by finding correlation of all the features wrt target variable i.e., price to see whether they are positively correlated or negatively correlated to find if they help in prediction process in model building process or not. But this is also one of the most important step as it also involves domain knowledge of the field of the data means you cannot simply remove the feature from your prediction process just because it is negatively correlated because it may contribute in future prediction for this you should take help of some domain knowledge personnel.

# ### correlation using Heatmap

# In[13]:


sns.set(font_scale=2.2)
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in data.iteritems():
   if type(colvalue[1]) == str:
        str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = data.columns.difference(str_list) 
# Create Dataframe containing only numerical features
house_num = data[num_list]
f, ax = plt.subplots(figsize=(35, 30))
plt.title('Correlation of features')
# Draw the heatmap using seaborn
#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
sns.heatmap(house_num.astype(float).corr(),linewidths=2.0,vmax=1.0, square=True, cmap="YlGnBu", linecolor='k', annot=True)
plt.show()


# ### Initial cleaning

# In[14]:


(data['bedrooms']> 5).value_counts().to_frame()


# In[15]:


(data['price']< 1000000).value_counts().to_frame()


# ### Filter to focus on  homes price under 1 Million?

# In[16]:


# Filter the dataset
midrange_homes = data[(data['price'] < 1000000) 
                         & (data['bedrooms'].isin(range(2, 6)))]

# View the first few rows
midrange_homes.head()


# In[17]:


midrange_homes.shape


# In[18]:


midrange_homes.describe()


# In[19]:


# Check for missing values by column
midrange_homes.isna().sum()


# ### Resolve missing values

# In[20]:


# View value counts for 'waterfront'
midrange_homes['waterfront'].value_counts()


# In[21]:


# Print medians of homes with and without 'waterfront'
print(midrange_homes[midrange_homes['waterfront'] == 1]['price'].median())
print(midrange_homes[midrange_homes['waterfront'] == 0]['price'].median())


# In[22]:


# Fill NaNs with 0.0 because it is the mode
midrange_homes['waterfront'] = midrange_homes['waterfront'].fillna(0.0)
midrange_homes['waterfront'] = midrange_homes['waterfront'].astype('int64')
midrange_homes.info()


# ### view

# In[23]:


# Create a histogram of 'view' values
plt.figure(figsize=(10,6))
midrange_homes['view'].hist()
plt.title('Histogram of \'view\' values')
plt.xlabel('\'view\' values')
plt.ylabel('Count')
plt.show();


# In[24]:


# Fill NaNs with 0.0 and check that missing `view` values are now resolved
midrange_homes['view'] = midrange_homes['view'].fillna(0.0).astype('int64')
midrange_homes.info()


# ### yr_renovated

# In[25]:


# Create a histogram of 'yr_renovated' values
plt.figure(figsize=(10,6))
midrange_homes['yr_renovated'].hist()
plt.title('Histogram of \'yr_renovated\' values')
plt.xlabel('\'yr_renovated\' values')
plt.ylabel('Count')
plt.show();


# In[26]:


midrange_homes['yr_renovated'].value_counts().to_frame()


# Here we can see in yr_renovated columns alot of years data are misiing it is better to delete coulumn.

# ### Drop unneeded columns

# In[27]:


# Drop unneeded columns
midrange_homes.drop(['id', 'date', 'sqft_above', 'yr_renovated','sqft_basement'], 
                    axis=1, inplace=True)

# Review the remaining columns
midrange_homes.info()


# ### After cleaning again checking correlation between features.

# In[28]:


# Create the correlation heatmap
data_for_scatter_matrix = midrange_homes.drop(['price'], axis=1)

plt.figure(figsize=(16,10))
sns.heatmap(data_for_scatter_matrix.corr(), center=0)
plt.title('Heatmap showing correlations between independent variables', 
          fontsize=18)
plt.show();


# In[29]:



# Check any number of columns with NaN or missing values 
print(midrange_homes.isnull().any().sum(), ' / ', len(midrange_homes.columns))
# Check any number of data points with NaN
print(midrange_homes.isnull().any(axis=1).sum(), ' / ', len(midrange_homes))


# In[30]:


midrange_homes.isna().sum()


# In[31]:


midrange_homes.head()


# In[32]:


from scipy.stats import pearsonr


# In[33]:


features = midrange_homes.iloc[:,1:].columns.tolist()
target = midrange_homes.iloc[:,0].name
features


# In[34]:


type(target)


# In[35]:


(features)


# In[36]:



# Finding Correlation of price woth other variables to see how many variables are strongly correlated with price
correlations = {}
for f in features:
    data_temp = midrange_homes[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]


# In[37]:


# Printing all the correlated features value with respect to price which is target variable
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]


# # STEP 4 : EDA or DATA VISUALIZATION
# 
# This is also a very important step in your prediction process as it help you to get aware you about existing patterns in the data how it is relating to your target variables etc.

# ### (1)What is the relationship between grade and price?

# In[38]:


# Create boxplots to compare 'grade' and 'price'
plt.figure(figsize=(10,8))
sns.boxplot(midrange_homes['grade'], midrange_homes['price'], color='skyblue')
plt.title('Distributions of prices for each grade', fontsize=18)
plt.xlabel('Grade')
plt.ylabel('Price (USD)')
plt.yticks([0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000],
            ['0', '100k', '200k', '300k', '400k', '500k', '600k',  '700k', '800k', '900k', '1M'])

plt.show();


# It looks like there could be substantial differences in price based on the grade of a house. For instance, only the outliers of grade-5 houses fall within the price range of grade-11 houses.
# 
# Let's make a direct comparison between the median prices of grade-7 and grade-10 homes:

# In[39]:


grade_7_med = midrange_homes[midrange_homes['grade'] == 7]['price'].median()
grade_10_med = midrange_homes[midrange_homes['grade'] == 10]['price'].median()

grade_10_med - grade_7_med


# There is a huge difference (almost $420500.0) between the median prices of grade-7 and grade-10 homes. Improving the grade of a home by that much is probably outside the reach of most homeowners. What if a homeowner could improve the grade of their home from 7 to 8?

# In[40]:


grade_8_med = midrange_homes[midrange_homes['grade'] == 8]['price'].median()

grade_8_med - grade_7_med


# Based on the boxplots above, we can see that the jump in median price from grade 7 to grade 8 is a big one, but if a homeowner could manage it, it could pay off. The median price of a grade-8 home is $125500.0 higher than the median price of a grade-7 home. Again, this is without considering any other factors, like the size or condition of these homes.

# ### What is the relationship between bedrooms and price?

# In[41]:


# Create boxplots for 'bedrooms' v. 'price'
plt.figure(figsize=(10,8))
sns.boxplot(midrange_homes['bedrooms'], midrange_homes['price'], color='skyblue')
plt.title('Distributions of prices for each number of bedrooms', fontsize=18)
plt.xlabel('Number of bedrooms')
plt.ylabel('Price (USD)')
plt.yticks([0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000],
            ['0', '100k', '200k', '300k', '400k', '500k', '600k',  '700k', '800k', '900k', '1M'])
plt.show();


# In[42]:


# Calculate percent differences in median prices
medians = []

for n in range(2,6):
    medians.append(midrange_homes[midrange_homes['bedrooms'] == n]['price'].median())

percent_differences = []
for m in range(0,len(medians)-1):
    percent_differences.append(round(((medians[m+1] - medians[m]) / medians[m]),2))
    
percent_differences


# The biggest difference in median price is between four and three bedrooms, where there is an increase of 22%

# ### What is the relationship between floors and price?

# In[43]:



var = 'floors'
data = pd.concat([ midrange_homes['price'],  midrange_homes[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 20))
fig = sns.boxplot(x=var, y="price", data=data)
fig.axis(ymin=0, ymax=1000000);


# In[44]:


# # Create boxplots for 'floors' v. 'price'
# plt.figure(figsize=(10,8))
# sns.boxplot(midrange_homes['floors'], midrange_homes['price'], color='skyblue')
# plt.title('Distributions of prices for each number of floors', fontsize=18)
# plt.xlabel('Number of floors')
# plt.ylabel('Price (USD)')         
# plt.yticks([0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000],
#             ['0', '100k', '200k', '300k', '400k', '500k', '600k',  '700k', '800k', '900k', '1M'])
# plt.show();


# What is the relationship between bathrooms and price?

# In[45]:


var = 'bathrooms'
data = pd.concat([midrange_homes['price'], midrange_homes[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 20))
fig = sns.boxplot(x=var, y="price", data=data)
fig.axis(ymin=0, ymax=1000000);


# # Where are the midrange homes in King County?

# In[46]:


# Define a function to create map-like scatter plots with color code
def locate_it(data, latitude, longitude, feature):
    
    """Create a scatterplot from lat/long data with color code.
    Parameters:
        data: a DataFrame
        latitude: the name of the column in your DataFrame that contains
            the latitude values. Pass this as a string.
        longitude: the name of the column in your DataFrame that contains
            the longitude values. Pass this as a string.
        feature: the name of the column whose values you want to use as 
            the values for your color code. Pass this as a string.
    Dependencies: matplotlib
    Returns: scatterplot"""
    
    plt.figure(figsize=(16,12))
    cmap = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(data[longitude], data[latitude], 
                     c=data[feature], vmin=min(data[feature]), 
                     vmax=max(data[feature]), alpha=0.5, s=5, cmap=cmap)
    plt.colorbar(sc)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('House {} by location'.format(feature), fontsize=18)
    plt.show();


# In[47]:


# Call locate_it for price by location
locate_it(midrange_homes, 'lat', 'long', 'price')


# In[48]:


# Call locate_it for sqft_living by location
locate_it(midrange_homes, 'lat', 'long', 'sqft_living')


# In[49]:


# Customize the plot for sqft_living by location
plt.figure(figsize=(16,12))
cmap = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(midrange_homes['long'], midrange_homes['lat'], 
                 c=midrange_homes['sqft_living'], 
                 vmin=min(midrange_homes['sqft_living']), 
                 vmax=np.percentile(midrange_homes['sqft_living'], 90), 
                 alpha=0.5, s=5, cmap=cmap)
plt.colorbar(sc)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('House square footage by location\n(Darkest blue = 90th percentile of size)', fontsize=14)
plt.show();


# In[50]:


# Call locate_it for grade by location
locate_it(midrange_homes, 'lat', 'long', 'grade')


# In[51]:


locate_it(midrange_homes, 'lat', 'long', 'condition')


# In[52]:


list(features)


# In[53]:


target


# In[54]:


feature_matrix = midrange_homes[features]
#feature_matrix = preprocessing.scale(feature_matrix)
feature_matrix_unscaled = midrange_homes[features]
lable_vector = midrange_homes['price']
feature_matrix_unscaled.head()
#feature_matrix[0::1000]


# In[55]:


style.use('fivethirtyeight')
cm = plt.cm.get_cmap('RdYlBu')
xy = range(19648)
z = xy
for feature in feature_matrix_unscaled:
    sc = plt.scatter(midrange_homes[feature], midrange_homes['price'], label = feature, c = z, marker = 'o', s = 30, cmap = cm)
    plt.colorbar(sc)
    plt.xlabel(''+feature)
    plt.ylabel('price')
    plt.yticks([0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000],
            ['0', '100k', '200k', '300k', '400k', '500k', '600k',  '700k', '800k', '900k', '1M'])
    plt.legend()
    plt.show()


# # Final preprocessing

# In[56]:


# Generate dummy variables
zip_dummies = pd.get_dummies(midrange_homes['zipcode'], prefix='zip')


# In[57]:


# Drop the original 'zipcode' column
mh_no_zips = midrange_homes.drop('zipcode', axis=1)

# Concatenate the dummies to the copy of 'midrange_homes'
mh_zips_encoded = pd.concat([mh_no_zips, zip_dummies], axis=1)

# Preview the new DataFrame
mh_zips_encoded.head()


# In[58]:


# Drop one of the dummy variables
mh_zips_encoded.drop('zip_98168', axis=1, inplace=True)

# Check the head again
mh_zips_encoded.head()


# # Creating the model

# In[59]:


# Import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[60]:


# Split the cleaned data into features and target
mh_features = midrange_homes.drop(['price'], axis=1)
mh_target = midrange_homes['price']


# In[61]:


# Define a function to run OLS and return model summary
def model_it(data, features):
    
    """Fit an OLS model and return model summary
    data: a DataFrame containing both features and target
    features: identical to 'data', but with the target dropped"""
    
    features_sum = '+'.join(features.columns)
    formula = 'price' + '~' + features_sum

    model = ols(formula=formula, data=data).fit()
    return model.summary()


# In[62]:


# Fit the model and return summary
model_it(midrange_homes, mh_features)


# # Second model

# In[63]:


# Drop unwanted features and rerun the modeling function
mh_features_fewer = mh_features.drop(['long', 'sqft_lot15'], axis=1)

model_it(midrange_homes, mh_features_fewer)


# # Third model

# In[64]:


# Split the data into features and target
mh_zips_encoded_features = mh_zips_encoded.drop(['price'], axis=1)
mh_zips_encoded_target = mh_zips_encoded['price']


# In[65]:


model_it(mh_zips_encoded, mh_zips_encoded_features)


# In[66]:


mh_zips_encoded_features.head()


# In[67]:


final_data = mh_zips_encoded_features.drop(['long','sqft_lot15','zip_98001','zip_98002','zip_98003','zip_98019','zip_98022','zip_98030'], axis =1)


# In[68]:


f_data =final_data.drop(['zip_98031','zip_98042','zip_98055','zip_98058','zip_98092','zip_98178','zip_98188','zip_98198'], axis = 1)


# In[69]:


f_data.head()


# In[70]:


X = f_data
y = mh_zips_encoded['price']
Xconst = sm.add_constant(X)
model = sm.OLS(y, Xconst, hasconst= True)
fitted_model = model.fit()
fitted_model.summary()


# In[71]:


X = f_data
y = mh_zips_encoded['price']
# Xconst = sm.add_constant(X)
model = sm.OLS(y, X)
fitted_model = model.fit()
fitted_model.summary()


# 
# Conclusion
# 
# So, we have seen that accuracy of OLS is around 82.9%. 
