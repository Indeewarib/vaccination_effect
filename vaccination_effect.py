#===================
# Import Libraries
#===================
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#===================
# Functions
#===================

#Function to visualize text in webpage
def write_text(text):
    st.text(text)

#Function to visualize general content in webpage
def write_content(text):
    st.write(text)

#Function to visualize data frames in webpage
def write_df(df):
    st.write(df)

#Function to include titles in webpage
def add_title(text):
    st.title(text)

#Function to include headers in webpage
def add_header(text):
    st.header(text)

#Function to include sub headers in webpage
def add_subheader(text):
    st.subheader(text)

#Function to include markdown codes in webpage
def add_markdown(text):
    st.markdown(text)

#Function to include captions in webpage
def add_captions(text):
    st.caption(text)


#Function to include headers in webpage
def add_header(text):
    st.header(text)

#===================
# Main
#===================

#---------------------------Read and explore datasets-----------------------------------------------------
full_path = os.path.realpath(__file__)
project_folder = os.path.dirname(full_path) 

infectious_diseases_df = pd.read_csv(f'{project_folder}\\data\\number_of_cases_of_infectious_diseases.csv')
vaccination_df = pd.read_csv(f'{project_folder}\\data\\number_of_vaccinated_one_year_olds.csv')

infectious_diseases_df.info()
infectious_diseases_df.describe()
infectious_diseases_df.head()

vaccination_df.info()
vaccination_df.describe()
vaccination_df.head()

#---------------------------Clean datasets----------------------------------------------------------------

#Drop the 'Code' columns from both dataframes
infectious_diseases_df.drop(['Code'], axis = 1, inplace = True)
vaccination_df.drop(['Code'], axis = 1, inplace = True)

#Change column names in both dataframes
infectious_diseases_df.columns = ['country','year','yaws_cases','polio_cases','guinea_worm_disease_cases','rabies_cases','malaria_cases','HIV_AIDS_cases','tuberculosis_cases','smallpox_cases','cholera_cases']
vaccination_df.columns = ['country','year','HeoB3','DTP','polio','population','measles','Hib3','rubella','rotavirus','BCG']

#In both dataframes, fill missing values with 0
infectious_diseases_df.fillna(0, inplace = True)
vaccination_df.fillna(0, inplace = True)

#Check dataframes
infectious_diseases_df.info()
infectious_diseases_df.head()

vaccination_df.info()
vaccination_df.head()

#---------------------------Further exploration with plots------------------------------------------------
#Plot number of cases of  each disease in each year using bar plot
for i in infectious_diseases_df.columns[2:]:
  fig, ax = plt.subplots(figsize=(12, 5))
  plt.title(i)

  sns.barplot(data=infectious_diseases_df, x = 'year', y=i, ci = None)
  plt.xticks(rotation = 90)
  plt.xlabel('Year', size = 2)
  plt.ylabel('Number of Cases')
  plt.show()

#Plot number of cases of  each disease in each year using choropleth 
#----made a copy from 'infectious_diseases_df' to use in choropleth map
infectious_diseases_df_copy = infectious_diseases_df.copy()

#----sort the copy of the dataframe by 'year' and 'country'
infectious_diseases_df_copy.sort_values(
    ["year", "country"], axis=0,
    ascending=True, inplace=True
)

#----plot the choropleth
for i in infectious_diseases_df_copy.columns[2:]:
   fig = px.choropleth(infectious_diseases_df_copy,locations='country', locationmode='country names',
                       color = i,hover_name="country", animation_frame="year",
                       title = f'{i} - Choropleth', color_continuous_scale='Viridis_r')
   fig.show()

#Plot vaccinations rate over year using line plot
for i in vaccination_df.columns[3:]:
  fig, ax = plt.subplots(figsize=(10, 4))
  plt.title(i)

  sns.lineplot(data=vaccination_df, x = 'year', y=i, ci = None)
  plt.xticks(rotation = 90)
  plt.xlabel('Year', size = 2)
  plt.ylabel('Vaccination_rate')
  plt.show()

#Show the correlation between vaccination rates and disease cases using heat map
#----merge the two tables
merged_vaccine_disease_df = pd.merge(infectious_diseases_df, vaccination_df, on=['country','year'])

#----plot heat map
plt.figure(figsize=(15, 10))
sns.heatmap(merged_vaccine_disease_df.corr(),annot = True)

#Show relationships between some vaccination rates and disease cases using scatter plots
fig, ax = plt.subplots(2,2,figsize = (10,8))
ax[0,0].scatter(merged_vaccine_disease_df.BCG,merged_vaccine_disease_df.tuberculosis_cases)
ax[0,0].set_xlabel('BCG')
ax[0,0].set_ylabel('tuberculosis_cases')

ax[0,1].scatter(merged_vaccine_disease_df.HIV_AIDS_cases,merged_vaccine_disease_df.measles)
ax[0,1].set_xlabel('measles')
ax[0,1].set_ylabel('HIV_AIDS_cases')

ax[1,0].scatter(merged_vaccine_disease_df.BCG,merged_vaccine_disease_df.HIV_AIDS_cases)
ax[1,0].set_xlabel('BCG')
ax[1,0].set_ylabel('HIV_AIDS_cases')

ax[1,1].scatter(merged_vaccine_disease_df.measles,merged_vaccine_disease_df.tuberculosis_cases)
ax[1,1].set_xlabel('measles')
ax[1,1].set_ylabel('tuberculosis_cases')

plt.show()
