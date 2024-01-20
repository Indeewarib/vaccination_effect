#===================
# Import Libraries
#===================
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
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
sns.heatmap(merged_vaccine_disease_df.drop('country', axis = 1).corr(), annot = True)

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

#---------------------------Before modeling--------------------------------------------------------
#Make a new copy of 'infectious_diseases_df'
infectious_diseases_new_copy_df = infectious_diseases_df.copy()

#Make a new copy of 'vaccination_df'
vaccination_df_new_copy_df = vaccination_df.copy()

#Change each number of vaccinations as a rate of relevant population
vaccine_columns = ['HeoB3', 'DTP', 'polio', 'measles', 'Hib3', 'rubella', 'rotavirus', 'BCG']

for column in vaccine_columns:
    vaccination_df_new_copy_df[column] = vaccination_df_new_copy_df[column] / vaccination_df_new_copy_df['population'] 

vaccination_df_new_copy_df.head(1)

#Scale down the number of each disease cases to a normal rate using 'MinMaxScaler in preprocessing'
columns_to_normalize = infectious_diseases_new_copy_df.columns[2:]
scaler = MinMaxScaler()
infectious_diseases_new_copy_df[columns_to_normalize] = scaler.fit_transform(infectious_diseases_new_copy_df[columns_to_normalize])

#Add a new column to 'infectious_diseases_new_copy_df' with sum of all diseases in a year
infectious_diseases_new_copy_df['total_disease_rate'] = infectious_diseases_new_copy_df[['yaws_cases', 'polio_cases',
       'guinea_worm_disease_cases', 'rabies_cases', 'malaria_cases',
       'HIV_AIDS_cases', 'tuberculosis_cases', 'smallpox_cases',
       'cholera_cases']].sum(axis=1)

infectious_diseases_new_copy_df.head(1)

#Merge 'infectious_diseases_new_copy_df' and 'vaccination_df_new_copy_df'
merged_scale_down_df = pd.merge(infectious_diseases_new_copy_df, vaccination_df_new_copy_df, on=['country','year'])

#Drop disease cases from 'merged_scale_down_df'
merged_scale_down_df.drop(['yaws_cases', 'polio_cases',
       'guinea_worm_disease_cases', 'rabies_cases', 'malaria_cases',
       'HIV_AIDS_cases', 'tuberculosis_cases', 'smallpox_cases',
       'cholera_cases'], axis = 1, inplace = True)

#Group 'merged_scale_down_df' to have data relevant to each year
disease_vaccine_scale_down_df = merged_scale_down_df.groupby('year').sum().reset_index()

#Drop 'population' column
disease_vaccine_scale_down_df.drop(['population','country'], axis = 1, inplace = True)

#---------------------------Modeling-------------------------------------------------------------------
#Set X values as the values of each vaccine rates column and drop the 'total_disease_rate'
X = disease_vaccine_scale_down_df.drop(['total_disease_rate'], axis=1) 

#Define the target variable y by setting 'total_disease_rate' 
y = disease_vaccine_scale_down_df['total_disease_rate']  

#Two lists to store each Mean Squared Error and Squared Error relevant to each random state
accuracies = []
accuracies_ = []

#Split the data into train and test sets and check them under five different random states
for random_state in [1, 23, 42, 15, 56]:
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)

#Initialize and train the model
  model = RandomForestRegressor() 

#Fit the model to train sets
  model.fit(X_train, y_train)

#Get predicted values for y
  y_predictions = model.predict(X_test)

#Evaluate the model
  accuracies.append(mean_squared_error(y_test, y_predictions))
  accuracies_.append(r2_score(y_test, y_predictions))

#Display the lists of Mean Squared Errors and Squared Error
print('Mean Squared Errors:', accuracies)
print(' Squared Error:', accuracies_)