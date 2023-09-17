#!/usr/bin/env python
# coding: utf-8

# In[40]:



 #Project Scenario:
    
    #You are a Data Scientist with a housing agency in Boston MA, you have been given access to a previous dataset on housing
    #prices derived from the U.S. Census Service to present insights to higher management. Based on your experience in Statistics,
    #what information can you provide them to help with making an informed decision? Upper management will like to get some insight
    #into the following.

  #get_ipython().run_line_magic('pinfo', 'not')

#Q. Is there a difference in median values of houses of each proportion of owner-occupied units built before 1940?

#Q. Can we conclude that there is no relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town?

#Q. What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner-occupied homes?


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm



# In[2]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df = pd.read_csv(boston_url)


# In[3]:


boston_df.describe()


# In[4]:


boston_df.head(10)


# In[7]:


ax = sns.boxplot(y = 'MEDV', data = boston_df)
ax.set_title('Owner-occupied homes')
plt.show()


# In[24]:


division_eval = boston_df.groupby('CHAS')[['MEDV']].mean().reset_index()
sns.set(style="whitegrid")
ax = sns.barplot(x="CHAS", y="MEDV", data=division_eval)
plt.show()


# In[25]:


boston_df.loc[(boston_df['AGE'] <= 35), 'Age_Group'] = '35 years and younger'
boston_df.loc[(boston_df['AGE'] > 35) & (boston_df['AGE'] < 70), 'Age_Group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df['AGE'] >= 70), 'Age_Group'] = '70 years and older'
ax3 = sns.boxplot(x = 'MEDV', y = 'Age_Group', data = boston_df)
ax3.set_title('Median value of owner-occupied homes per Age Group')


# In[27]:


ax4 = sns.scatterplot(y = 'NOX', x = 'INDUS', data = boston_df)
ax4.set_title('Nitric oxide concentration per proportion of non-retail business acres per town')


# In[29]:


sns.catplot(x='PTRATIO', kind = 'count' ,data = boston_df)
plt.show()


# In[30]:


scipy.stats.ttest_ind(boston_df[boston_df['CHAS'] == 0]['MEDV'],
                   boston_df[boston_df['CHAS'] == 1]['MEDV'], equal_var = True)


# In[33]:


boston_df["AGE"].value_counts()


# In[36]:


boston_df.loc[(boston_df["AGE"] <= 35),'age_group'] = '35 years and younger'
boston_df.loc[(boston_df["AGE"] > 35) & (boston_df["AGE"]<70),'age_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df["AGE"] >= 70),'age_group'] = '70 years and older'
boston_df


# In[37]:


low = boston_df[boston_df["age_group"] == '35 years and younger']["MEDV"]
mid = boston_df[boston_df["age_group"] == 'between 35 and 70 years']["MEDV"]
high = boston_df[boston_df["age_group"] == '70 years and older']["MEDV"]

f_statistic, p_value = scipy.stats.f_oneway(low, mid, high)
print("F_Statistic: {0}, P-Value: {1}".format(f_statistic,p_value))


# In[38]:


ax = sns.scatterplot(x="NOX", y="INDUS", data=boston_df)
scipy.stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])


# In[39]:


x = boston_df['DIS']
y = boston_df['MEDV']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predisction = model.predict(x)

model.summary()


# In[ ]:




