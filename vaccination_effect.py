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

#===================
# Main
#===================

#Initializing web page
add_title('Vaccination Effect')
add_subheader('The Impact of Basic Vaccinations on Infectious Disease Spread')
write_content('''This project aims to investigate the potential impact of basic vaccinations, such as polio 
                for one-year-olds, on the spread of infectious diseases. The ultimate goal is to 
                build a model that can predict the effect of basic vaccinations on the spread of infectious diseases and 
                potentially anticipate the impact of vaccination data on future diseases, whether it has an effect on controlling them.''')
add_markdown('**Check dataset**:')
add_markdown("[Click here](https://www.kaggle.com/datasets/imtkaggleteam/pandemics-in-world?select=2-+the-worlds-number-of-vaccinated-one-year-olds.csv)")
write_content('For more information read ```README.md``` file')

st.sidebar.subheader('Settings')
write_content('Check **sidebar menu** for more advance features')

#---------------------------Read and explore datasets-----------------------------------------------------
full_path = os.path.realpath(__file__)
project_folder = os.path.dirname(full_path) 

infectious_diseases_df = pd.read_csv(f'{project_folder}\\data\\number_of_cases_of_infectious_diseases.csv')
vaccination_df = pd.read_csv(f'{project_folder}\\data\\number_of_vaccinated_one_year_olds.csv')

#Include information of raw datasets in side menu to check if need
if st.sidebar.checkbox('Explore raw datasets'):
    add_markdown(':orange[Explore raw infectious diseases dataset]')
    with st.expander('Show infectious diseases dataset exploration'):
        add_markdown(':red[info()]')
        write_content(infectious_diseases_df.info())
        add_markdown(':red[description()]')
        write_content(infectious_diseases_df.describe())
        add_markdown(':red[head()]')
        write_content(infectious_diseases_df.head())

    add_markdown(':orange[Explore raw vaccinations dataset]')
    with st.expander('Show vaccinations dataset exploration'):
        add_markdown(':red[info()]')
        write_content(vaccination_df.info())
        add_markdown(':red[description()]')
        write_content(vaccination_df.describe())
        add_markdown(':red[head()]')
        write_content(vaccination_df.head())

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
#Show the correlation between vaccination rates and disease cases using heat map
#----merge the two tables
merged_vaccine_disease_df = pd.merge(infectious_diseases_df, vaccination_df, on=['country','year'])

if st.sidebar.checkbox('Check correlation'):
    add_markdown(':orange[The correlation among vaccine rates and disease cases]')
    with st.expander('Show heatmap to check correlation among vaccines and disease rates'):
#----plot heat map
        fig = plt.figure(figsize=(15, 10))
        sns.heatmap(merged_vaccine_disease_df.drop('country', axis = 1).corr(), annot = True)
        st.pyplot(fig)
        add_captions(f'Correlation among diseases and vaccine rates')

#Visualize plots in webpage
add_header('Visualize plots')
write_content('Here includes all plots used to explore the datasets')

with st.expander('Show bar plots of number of cases of  each disease recorded each year'):
#Plot number of cases of  each disease in each year using bar plot
    for i in infectious_diseases_df.columns[2:]:
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.title(i)

        sns.barplot(data=infectious_diseases_df, x = 'year', y=i, ci = None)
        plt.xticks(rotation = 90)
        plt.xlabel('Year', size = 2)
        plt.ylabel('Number of Cases')
        plt.show()
        st.pyplot(fig)
        add_captions(f'Number of {i} each year')

#Plot number of cases of  each disease in each year using choropleth 
#----made a copy from 'infectious_diseases_df' to use in choropleth map
infectious_diseases_df_copy = infectious_diseases_df.copy()

#----sort the copy of the dataframe by 'year' and 'country'
infectious_diseases_df_copy.sort_values(
    ["year", "country"], axis=0,
    ascending=True, inplace=True
)

#Visualize choropleth maps
with st.expander('Show choropleth maps of each disease cases changing over each year'):
#Include a check box to check choropleth map when needs
    if st.checkbox('Check choropleth maps of each disease'):
#----plot the choropleth
        for i in infectious_diseases_df_copy.columns[2:]:
            fig = px.choropleth(infectious_diseases_df_copy,locations='country', locationmode='country names',
                       color = i,hover_name="country", animation_frame="year",
                       title = f'{i} - Choropleth', color_continuous_scale='Viridis_r')
            fig.show()

#Visualize line plots
with st.expander('Show line plots of vaccinations rate over years'):
#Plot vaccinations rate over year using line plot
    for i in vaccination_df.columns[3:]:
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.title(i)

        sns.lineplot(data=vaccination_df, x = 'year', y=i, ci = None)
        plt.xticks(rotation = 90)
        plt.xlabel('Year', size = 2)
        plt.ylabel('Vaccination_rate')
        plt.show()
        st.pyplot(fig)
        add_captions(f'Change of {i} rates each year')

with st.expander('Show scatter plots'):
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
    st.pyplot(fig)
    add_captions('Relationship between BCG,measles vaccine rates and HIV,tuberculosis disease cases')

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

#Group 'merged_scale_down_df' to have data relevant to each year
disease_vaccine_scale_down_df = merged_scale_down_df.groupby('year').sum().reset_index()

#Drop 'population' column
disease_vaccine_scale_down_df.drop(['population','country'], axis = 1, inplace = True)

#Add a checkbox to sidebar menu to check final datasets if needed
if st.sidebar.checkbox('Display final datasets'):
    add_header('Final datasets')

#Add an expander with each dataset to check how the dataset came
    add_subheader('Final scaled down dataset')
    write_content('Includes data on each vaccine rate and total infectious disease cases in each year')
    write_df(disease_vaccine_scale_down_df)
    with st.expander('Show steps done'):
      write_content('Step 01: Scaled down ```infectious_diseases_df``` using **MinMaxScaler**')
      write_content('Step 02: Change each vaccine rate as a rate of **Population** in ```vaccination_df```')
      write_content('Step 03: Add the column **total_disease_rate** including sum of all disease cases reported each year')
      write_content('Step 04: Mereged ```infectious_diseases_df``` and ```vaccination_df``` on **country** and **year** and drop all disease cases columns')
      write_content('Step 05: Grouped ```disease_vaccine_scale_down_df``` by **year**')  

    add_subheader('Infectious diseases dataset')
    write_content('Includes data of infectious disease cases in different countries and year')
    write_df(infectious_diseases_df)
    with st.expander('Show steps done'):
      write_content('Step 01: Droped **Code** column')
      write_content('Step 02: Replaced NaN values with zero')

    add_subheader('Vaccination rates dataset')
    write_content('Includes data of vaccine rates of one year olds in different countries and year')
    write_df(vaccination_df)
    with st.expander('Show steps done'):
      write_content('Step 01: Droped **Code** column')
      write_content('Step 02: Replaced NaN values with zero')

#Add side bar to check more information by filtering
if st.sidebar.checkbox('Find informations on each year records'):

#Add a text to include what can be checked
    write_content('Check the total disease rate and each vaccine count of years')

#Using text input in the web page
    year_checks = st.text_input('Year', placeholder="1990")

#Creating regular expression from text written by year
    year_regexp = f'.*{year_checks}.*'

#Printing data filtered using regular expression
    disease_vaccine_scale_down_df['year'] = disease_vaccine_scale_down_df['year'].astype(str)
    write_df(disease_vaccine_scale_down_df[disease_vaccine_scale_down_df['year'].str.match(year_regexp)])


#---------------------------Modeling-------------------------------------------------------------------
add_header('Modeling - machine learning application')
with st.expander('Show model'):
    st.subheader('The model predicting the effect of vaccination rates on total disease cases')
   
#Initialize and train the model
    model = RandomForestRegressor()  

    choices_x = st.multiselect('Select the vaccine/vaccines to construct the model:', ["HeoB3", "DTP", "polio",
                                                                            "measles", "Hib3", "rubella","rotavirus","BCG"])

    choices_y = st.multiselect('Select the disease case/cases to construct the model:', ["yaws_cases", "polio_cases","guinea_worm_disease_cases","rabies_cases","malaria_cases",
                                                                            "HIV_AIDS_cases", "tuberculosis_cases", "smallpox_cases",
                                                                            "cholera_cases","total_disease_rate"])

    test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)

    if len(choices_x) > 0 and len(choices_x) > 0 and st.button('RUN MODEL'):
        with st.spinner('Training...'):

#Set X and y values 
            X = disease_vaccine_scale_down_df[choices_x]
            y = disease_vaccine_scale_down_df[choices_y]  

#Two lists to store each Mean Squared Error and Squared Error relevant to each random state
            accuracies = []
            accuracies_ = []

#Split the data into train and test sets and check them under five different random states
            for random_state in [1, 23, 42, 15, 56]:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)

#Fit the model to train sets
                model.fit(X_train[choices_x] , y_train[choices_y])

#Get predicted values for y
                y_predictions = model.predict(X_test)

#Evaluate the model
                accuracies.append(mean_squared_error(y_test, y_predictions))
                accuracies_.append(r2_score(y_test, y_predictions))

#Display the lists of Mean Squared Errors and Squared Error
            st.write('Mean Squared Errors:', accuracies)
            st.write(' Squared Error:', accuracies_)