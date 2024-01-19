#===================
# Import Libraries
#===================
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

#Read and check datasets
infectious_diseases_df = pd.read_csv("number_of_cases_of_infectious_diseases.csv")
vaccination_df = pd.read_csv("number_of_vaccinated_one_year_olds.csv")

infectious_diseases_df.info()
infectious_diseases_df.describe()
infectious_diseases_df.head()

vaccination_df.info()
vaccination_df.describe()
vaccination_df.head()