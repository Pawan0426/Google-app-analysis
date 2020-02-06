#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In this mini project, you will explore and analyze the Google Apps dataset. There is no question so you will have to be self motivated in finding out what you want to know about this data, what looks interesting to you and what results to want to communicate.
# 
# We will provide some questions and examples as initial guidance. You can (and should) also discuss with the tutors and other class mates on ideas, things you want to achieve and Python techniques during class.

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[115]:


apps = pd.read_csv('googleplaystore.csv')


# In[116]:


apps= apps.dropna()


# In[ ]:





# In[49]:


apps.head()


# ### One of the greatest qualities of a data analyst is the ability to ask questions yourself. Be curious about your data.
# 
# Below is a suggested workflow and some questions to motivate you. Try to think of other questions as well.

# 1- Inspect the data, calculate some statistics and write some comments on interesting observations.
# 
# 
# 2- Data cleaning:
#     - Use string slicing and string methods to change columns like 'Size', 'Installs', 'Price', ... to numbers
#     - Remove duplicate rows
# 
# 
# 3- Visualize and comment on some or all columns. For example:
#     - Which categories have the most apps? Which categories are most popular (by installs)?
#     - What is the range of application size (maybe using a boxplot?)
#     - Visualize the distribution of price for paid apps only
#     - How many apps have multiple genres?
#     - Plot a bar chart for Last Updated by year
# 
# 
# 4- More possibly interesting questions to explore:
#     - Can you make a scatter plot to show the relationship between rating and number of installs?
#     - Do free apps have more installs than paid apps on average?
#     - Which categories appear the most among top 100 most expensive apps?
#     - Can you show the top 100 most common words that appear in apps name?
# 
# 5- Alternatively, you can also pick a category that you like and do in-depth analysis on that category. For example: Game apps
#     - Make a chart that shows the relationship between price, ratings and review.
#     - What are the most popular genres?
#     - Are games that support more android devices more popular?

# In[50]:


apps[apps.duplicated()]


# In[58]:


#cars[cars.Model.duplicated()]
#apps[apps.all.duplicated()]
#cars[~(cars.Model.duplicated())]
#apps[~(apps.All.diplicated())]
apps= apps.drop_duplicates()
apps


# In[118]:


apps.Size.unique()


# In[117]:


M_rows= apps["Size"].str.contains("M")
K_rows= apps["Size"].str.contains("k")
V_rows= apps["Size"].str.contains("Varies with device")

apps.loc[M_rows,"Size"]= apps.loc[M_rows, "Size"].str.replace("M","").astype("float")*1000000
apps.loc[K_rows,"Size"]= apps.loc[K_rows,"Size"].str.replace("k","").astype("float")*1000
apps.loc[V_rows,"Size"]= apps.loc[V_rows,"Size"].str.replace("Varies with device","0").astype("float")


# In[35]:


apps.Price.unique()


# In[42]:


#apps.Price = apps.Price.str.replace("$", "").astype(float)
print(apps.Price)


# In[44]:


apps.Installs= apps.Installs.str.replace("+","").str.replace(",","")
print(apps.Installs)


# In[18]:


apps.Category.unique()


# In[68]:


#visualize Which categories have the most apps? Which categories are most popular (by installs)?
apps.Category.value_counts().plot("bar")


# In[101]:


apps.Installs = apps.Installs.str.replace("+","").str.replace(",","").astype("float")
apps.Installs


# In[100]:


apps.info()


# In[103]:


#Which categories are most popular (by installs)?
gb= apps.groupby("Category")
gb["Installs"].sum().plot("bar", figsize=(12,6))


# In[119]:


#What is the range of application size (maybe using a boxplot?)
apps.Size.plot("bar")


# In[120]:


apps.Size.plot("box")


# ### Remember to comment something after you produce a statistic table or chart to summarize the results or discuss things that you found interesting.
# 
# Communication is also an important skill in data analysis. Without meaningful summary and comments, people won't understand your beautiful chart.

# <p style="color:green;">Example</p>

# Group the data by category and explore the average rating

# In[12]:


gb = apps.groupby('Category')

cat_mean = gb.mean()

cat_mean


# In[31]:


rating = cat_mean.plot(kind='bar', y='Rating', figsize=(20,6))

rating.set_ylim(3, 4.5)

rating.axhline(y=4, color='r')


# We can see that Events apps are usually rated very high in the market. Other highly rated categories are Art, Books and Education.
# 
# All categories except Dating apps have an average rating above 4. It could either be that people are often frustrated with dating apps, or that these apps tend to have poor design and functionality.

# ### Additionally, you can look for well made notebooks on kaggle.com. Here is one good example on the Titanic dataset that you can draw inspiration from:
# 
# https://www.kaggle.com/ash316/eda-to-prediction-dietanic
# 
# 

# ### Now it's your turn to explore the data

# In[ ]:




