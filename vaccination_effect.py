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

