##############################
#      Try new things        #
##############################

# What am I interested in?
# Is there a country with a sudden increase in female_completion --> e.g. due to change in equality laws
# Was there a consequent change in GDP/life expectancy/employment
    # we expect employment ot increase bc now women are more likely to work
    # likely gpd with also be increasing
    # not sure abot life expectancy

import numpy as np
import pandas as pd

female_completion_rate = pd.read_csv('female_completion_rate.csv', index_col='Country')
male_completion_rate = pd.read_csv('male_completion_rate.csv', index_col='Country')
life_expectancy = pd.read_csv('life_expectancy.csv', index_col='Country')
gdp_per_capita = pd.read_csv('gdp_per_capita.csv', index_col='Country')
employment_above_15 = pd.read_csv('employment_above_15.csv', index_col='Country')

print(female_completion_rate.index.values)

# Plot female completion by country

female_completion_country = female_completion_rate.loc['United States']
male_completion_country = male_completion_rate.loc['United States']
life_expectancy_country = life_expectancy.loc['United States']
gdp_country = gdp_per_capita.loc['United States']
employment_country = employment_above_15.loc['United States']

female_completion_country.plot()
male_completion_country.plot()
life_expectancy_country.plot()
gdp_country.plot()
employment_country.plot()

