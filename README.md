# Vaccination Effect

## Understanding the Impact of Basic Vaccinations on Infectious Disease Spread
### Overview
This project aims to investigate the potential impact of basic vaccinations, such as polio for one-year-olds, on the spread of infectious diseases. By analyzing vaccination rates and the total reported cases of infectious diseases over a specific time period, seek to understand whether there is a correlation between the two factors. The ultimate goal is to build a model that can predict the effect of basic vaccinations on the spread of infectious diseases and potentially anticipate the impact of vaccination data on future diseases, whether it has an effect on controlling them.

To launch the project you need to satisfy the requirements present in ```requirements.txt``` file and run the following command: ```streamlit run .\vaccination_effect.py```

### Data Sources
>From https://www.kaggle.com/datasets/imtkaggleteam/pandemics-in-world?select=2-+the-worlds-number-of-vaccinated-one-year-olds.csv

### Repository Structure
The project contains the following files:
- ```vaccination_effect.py```: python code to reach the main goal
- ```data``` folder: containing dataset CSV files
- ```test``` folder: python code to test project features and graphs
    - ```python_codes_raw.py```: file to test python commands
- ```requirements.txt```: required python libraries

### Useful information
- The ```number_of_vaccinated_one_year_olds``` dataset contains the rate of basic vaccinations given only to one year olds around the world

### Machine Learning
The main goal is to check the impact of basic vaccines have on spread of infectious diseases
The target variable is: ```total_disease_rate```
The variables affect affect include: ```['HeoB3', 'DTP', 'polio', 'measles', 'Hib3', 'rubella', 'rotavirus', 'BCG']```
The model used is: ```RandomForestRegressor() ``` (since data does not have a linear relationship)

### Future Considerations
Aim to expand the analysis to include additional infectious diseases and vaccination types. Furthermore, to explore the potential impact of emerging diseases on vaccination strategies and public health efforts.

