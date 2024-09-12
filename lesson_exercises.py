##################################################
####   Lesson 3: NumPy and Pandas 2D data    #####
##################################################

#####################################
#      1: Generating Questions      #
#####################################

# Q1: How many times did ppl enter/exit the turnstile
# Q2: Effect of weather on subway transport use
        # obvious weather factors: temp, rain, cloud, fog, wind, ...
        # non-obvious weather factors: air pressure,
# Q3: Travelling per weekday
        # E.g. observe weekdays vs. weekends
# Q4: Travelling per time of day
# Q5: Travelling per station
        # Are some more popular/busy
# Carolines Questions
# Q6: What variables are related to subway ridership?
    # Which stations have the most riders?
    # What are the ridership patterns over time?
    # How does the weather affect ridership?
# Q7: What patterns can I find in the weather?
    # Is the temperature rising throughout the month?
    # How does weather vary across the city?

#####################################
#            2: 2D data             #
#####################################

### Data with both rows & columns
### Python: list of lists
### Numpy: 2D array
### Pandas: Dataframe

# 2D arrays, as opposed to array of arrays:
    # more memory efficient
    # accessing elements is a bit different
        # a[1,3] rather than a[1][3]
    # mean(), std(), etc. operate on entire array

import numpy as np

# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

# Change False to True for each block of code to see what it does

# Accessing elements
if True:
    print(ridership[1, 3])
    print(ridership[1:3, 3:5]) # up to but NOT including 3/5
    print(ridership[1, :])
    
# Vectorized operations on rows or columns
if True:
    print(ridership[0, :] + ridership[1, :])
    print(ridership[:, 0] + ridership[:, 1])
    
# Vectorized operations on entire arrays
if True:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    print(a + b)
    
    # Fill in this function to find the station (column) with the maximum riders on the
    # first day (row 1), then return the mean riders per day for that station. Also
    # return the mean ridership overall for comparsion.

def mean_riders_for_max_station(ridership):
    max_station = ridership[0, ].argmax() # returns location of max ridership
    mean_for_max = ridership[:,max_station].mean() # mean riders for maximum station 
    overall_mean = ridership.mean() # mean riders per day
    return (overall_mean, mean_for_max)

mean_riders_for_max_station(ridership)

#####################################
#          3: NumPy axis            #
#####################################

### Choose axis to determine means/... for row or column
    #  axis = 1  ->  mean for each row
    #  axis = 0  ->  mean for each column
    
import numpy as np

# Change False to True for this block of code to see what it does

# NumPy axis argument
if True:
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    print(a.sum())
    print(a.sum(axis=0)) # column
    print(a.sum(axis=1)) # row
    
# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

    
def min_and_max_riders_per_day(ridership):
    daily_ridership = ridership.mean(axis=0) # average across days
    max_daily_ridership = daily_ridership.max()     
    min_daily_ridership = daily_ridership.min() 
    return (max_daily_ridership, min_daily_ridership)

min_and_max_riders_per_day(ridership)

#####################################
#        4: Panda Dataframes        #
#####################################

### Downside to NumPy 2D arrays: the data type for every datapoint has to be the same
### when entering different variables e.g. names, dates, values, ... it is
### not possible to calculate means/etc., it'll all get transformed to one type

### -> Panda dataframes allows for data of different types to be combined in one
    # columns have names
    # rows have indexes
### each column is assumed to be a different type
### when taking mean of whole dataframe
    # it's automatically done on columns
    # can be done on rows if your dataframe works like that
    # non-numerical data is ignored 

import pandas as pd

# Subway ridership for 5 stations on 10 different days
ridership_df = pd.DataFrame(
    data=[[   0,    0,    2,    5,    0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [  95,  229,  255,  496,  201],
          [   2,    0,    1,   27,    0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
)

# Change False to True for each block of code to see what it does

# DataFrame creation
if True:
    # You can create a DataFrame out of a dictionary mapping column names to values
    df_1 = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    print(df_1)

    # You can also use a list of lists or a 2D NumPy array
    df_2 = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=['A', 'B', 'C'])
    print(df_2)
   

# Accessing elements
if True:
    print(ridership_df.iloc[0])
    print(ridership_df.loc['05-05-11'])
    print(ridership_df['R003'])
    print(ridership_df.iloc[1, 3])
    
# Accessing multiple rows
if True:
    print(ridership_df.iloc[1:4])
    
# Accessing multiple columns
if True:
    print(ridership_df[['R003', 'R005']])
    
# Pandas axis
if True:
    df = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    print(df.sum())         # automatically calculates per column
    print(df.sum(axis=1))
    print(df.values.sum())  # to get total sum of ALL values in df

# Solutions (which worked different?)
def mean_riders_for_max_station(ridership):
    max_station = ridership.iloc[0].argmax()        # For her argmax returned 'R006', for me it returns '3'
    mean_for_max = ridership[max_station].mean()
    overall_mean = ridership.values.mean()
    return (overall_mean, mean_for_max)

def mean_riders_for_max_station2(ridership_df):
    max_station = ridership_df.iloc[0].idxmax()     # Now it returns 'R006'
    mean_for_max = ridership_df[max_station].mean()
    overall_mean = ridership_df.values.mean()
    return (overall_mean, mean_for_max)

mean_riders_for_max_station2(ridership_df)

#####################################
#    5: Calculating Correlation     #
#####################################

# Understand and Interpreting Correlations
# This page contains some scatterplots of variables with different values of correlation.
    # http://onlinestatbook.com/2/describing_bivariate_data/pearson.html
# This page lets you use a slider to change the correlation and see how the data might look.
    # http://rpsychologist.com/d3/correlation/
# Pearson's r only measures linear correlation! This image shows some different linear and non-linear relationships and what Pearson's r will be for those relationships.
    # https://en.wikipedia.org/wiki/Correlation_and_dependence#/media/File:Correlation_examples2.svg

# Corrected vs. Uncorrected Standard Deviation
    # By default, Pandas' std() function computes the standard deviation using Bessel's correction. Calling std(ddof=0) ensures that Bessel's correction will not be used.
    # https://en.wikipedia.org/wiki/Bessel%27s_correction
    
# Previous Exercise
    # The exercise where you used a simple heuristic to estimate correlation was the "Pandas Series" exercise in the previous lesson, "NumPy and Pandas for 1D Data".

# Pearson's r in NumPy
    # NumPy's corrcoef() function can be used to calculate Pearson's r, also known as the correlation coefficient.
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html


import pandas as pd

filename = 'nyc_subway_weather.csv'
subway_df = pd.read_csv(filename)

'''
Fill in this function to compute the correlation between the two
input variables. Each input is either a NumPy array or a Pandas
Series.
correlation = average of (x in standard units) times (y in standard units)
Remember to pass the argument "ddof=0" to the Pandas std() function!
'''

def correlation(x, y):
    mx = x.mean()
    sx = x.std(ddof=0)
    my = y.mean()
    sy = y.std(ddof=0)
    x_standardized = (x - mx)/sx
    y_standardized = (y - my)/sy
    product = x_standardized * y_standardized
    r = product.mean()
    return r

# More concise!
def correlation2(x,y):
    std_x = (x - x.mean()) / x.std(ddof=0)
    std_y = (y - y.mean()) / y.std(ddof=0)
    return (std_x * std_y).mean()

entries = subway_df['ENTRIESn_hourly']
cum_entries = subway_df['ENTRIESn']
rain = subway_df['meanprecipi']
temp = subway_df['meantempi']

print(correlation(entries, rain))
print(correlation(entries, temp))
print(correlation(rain, temp))

print(correlation(entries, cum_entries))

### Panda Axis Names
### instead of axis = 0 or axis = 1 you can use axis = 'index' or axis = 'columns'
    # can be tricky to remember which does what so just try it out

#####################################
#   6: DF vectorized operations     #
#####################################

### similar to vectorized operations for 2D NumPy arrays
### elements are matched up by index & column rather than position


import pandas as pd

# Examples of vectorized operations on DataFrames:
# Change False to True for each block of code to see what it does

# Adding DataFrames with the column names
if True:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90]})
    print(df1 + df2)
    
# Adding DataFrames with overlapping column names 
if True:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df2 = pd.DataFrame({'d': [10, 20, 30], 'c': [40, 50, 60], 'b': [70, 80, 90]})
    print(df1 + df2)

# Adding DataFrames with overlapping row indexes
if True:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]},
                       index=['row1', 'row2', 'row3'])
    df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90]},
                       index=['row4', 'row3', 'row2'])
    print(df1 + df2)

# --- Quiz ---
# Cumulative entries and exits for one station for a few hours.
entries_and_exits = pd.DataFrame({
    'ENTRIESn': [3144312, 3144335, 3144353, 3144424, 3144594,
                 3144808, 3144895, 3144905, 3144941, 3145094],
    'EXITSn': [1088151, 1088159, 1088177, 1088231, 1088275,
               1088317, 1088328, 1088331, 1088420, 1088753]
})

    
'''
Fill in this function to take a DataFrame with cumulative entries
and exits (entries in the first column, exits in the second) and
return a DataFrame with hourly entries and exits (entries in the
first column, exits in the second).
'''

def get_hourly_entries_and_exits(entries_and_exits):
    return entries_and_exits - entries_and_exits.shift()
get_hourly_entries_and_exits(entries_and_exits)

# Alternative Solution
# As an alternative to using vectorized operations, you could also use
# the code return entries_and_exits.diff() to calculate the answer in a single step.

def get_hourly_entries_and_exits2(entries_and_exits):
    return entries_and_exits.diff()
get_hourly_entries_and_exits2(entries_and_exits)


#####################################
#   7: non-build-in functions DFs   #
#####################################

# for panda series it was apply()
# but for dataframes apply() does something else!

# df.applymap(function)

import pandas as pd

# Change False to True for this block of code to see what it does

# DataFrame applymap()
if True:
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [10, 20, 30],
        'c': [5, 10, 15]
    })
    
    def add_one(x):
        return x + 1
        
    print(df.applymap(add_one))
    
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)
    
'''
Fill in this function to convert the given DataFrame of numerical
grades to letter grades. Return a new DataFrame with the converted
grade.

The conversion rule is:
    90-100 -> A
    80-89  -> B
    70-79  -> C
    60-69  -> D
    0-59   -> F
'''
grade = grades_df.loc['Andre','exam1']
def convert_grade(grade):
    if grade >= 90:
        return 'A'
    elif grade >= 80:
        return 'B'
    elif grade >= 70:
        return 'C'
    elif grade >= 60:
        return 'D'
    else:
        return 'F'

def convert_grades(grades):
    return grades.applymap(convert_grade)

convert_grades(grades_df)

#####################################
#        8: Dataframe apply()       #
#####################################

### applies your function to each column (or row) one by one?
### makes sense if values depend on each other? e.g. grading on a curve

import pandas as pd

grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

# Change False to True for this block of code to see what it does

# DataFrame apply()
if True:
    def convert_grades_curve(exam_grades):
        # Pandas has a bult-in function that will perform this calculation
        # This will give the bottom 0% to 10% of students the grade 'F',
        # 10% to 20% the grade 'D', and so on. You can read more about
        # the qcut() function here:
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
        return pd.qcut(exam_grades,
                       [0, 0.1, 0.2, 0.5, 0.8, 1],
                       labels=['F', 'D', 'C', 'B', 'A'])
        
    # qcut() operates on a list, array, or Series. This is the
    # result of running the function on a single column of the
    # DataFrame.
    print(convert_grades_curve(grades_df['exam1']))
    
    # qcut() does not work on DataFrames, but we can use apply()
    # to call the function on each column separately
    print(grades_df.apply(convert_grades_curve))

'''
Fill in this function to standardize each column of the given
DataFrame. To standardize a variable, convert each value to the
number of standard deviations it is above or below the mean.
'''
    
def standardize_column(column):
    return (column - column.mean()) / column.std()

standardize_column(grades_df['exam1'])

def standardize_df(df):
    return grades_df.apply(standardize_column)
    
standardize_df(grades_df)    

# Note: In order to get the proper computations, we should actually be setting
# the value of the "ddof" parameter to 0 in the .std() function.
# Note that the type of standard deviation calculated by default is different
# between numpy's .std() and pandas' .std() functions. By default, numpy calculates
# a population standard deviation, with "ddof = 0". On the other hand, pandas calculates
# a sample standard deviation, with "ddof = 1". If we know all of the scores, then we have a population
# - so to standardize using pandas, we need to set "ddof = 0".

### Other use of apply() for dataframes
    # call function on each columnn and just return a single value
    # apply will create a new series where df columns have been reduced to single value each
    # e.g. like with max but for custom functions (e.g. getting second max)

import numpy as np
import pandas as pd

df = pd.DataFrame({
    'a': [4, 5, 3, 1, 2],
    'b': [20, 10, 40, 50, 30],
    'c': [25, 20, 5, 15, 10]
})

# Change False to True for this block of code to see what it does

# DataFrame apply() - use case 2
if True:   
    print(df.apply(np.mean))
    print(df.apply(np.max))

'''
Fill in this function to return the second-largest value of each 
column of the input DataFrame.
'''

def second_largest_in_column(column):
    sorted_column = column.sort_values(ascending=False)
    return sorted_column.iloc[1]

second_largest_in_column(df["a"])

def second_largest(df):
    return df.apply(second_largest_in_column)

second_largest(df)

#####################################
#   9: Add DataFrame to a Series    #
#####################################

### 

import pandas as pd

# Change False to True for each block of code to see what it does

# Adding a Series to a square DataFrame
if True:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print(df)
    print('')# Create a blank line between outputs
    print(df + s)
    
# Adding a Series to a one-row DataFrame 
if True:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({0: [10], 1: [20], 2: [30], 3: [40]})
    
    print(df)
    print('') # Create a blank line between outputs
    print(df + s)

# Adding a Series to a one-column DataFrame
if True:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({0: [10, 20, 30, 40]})
    
    print(df)
    print('') # Create a blank line between outputs
    print(df + s)
    

    
# Adding when DataFrame column names match Series index
if True:
    s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })
    
    print(df)
    print('') # Create a blank line between outputs
    print(df + s)
    
# Adding when DataFrame column names don't match Series index
if True:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })
    
    print(df)
    print('') # Create a blank line between outputs
    print(df + s)
    
### alternatively
df.add(s) # does the same thing as df + s, BUT
df.add(s, axis ='index') # we can now change the series to be added to the rows, not columns

#####################################
#  10: Standardizing without apply  #
#####################################

import pandas as pd

# Adding using +
if True:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print(df)
    print('') # Create a blank line between outputs
    print(df + (s))
    
# Adding with axis='index'
if True:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print(df)
    print('') # Create a blank line between outputs
    print(df.add(s, axis='index'))
    # The functions sub(), mul(), and div() work similarly to add()
    
# Adding with axis='columns'
if True:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print(df)
    print('') # Create a blank line between outputs
    print(df.add(s, axis='columns'))
    # The functions sub(), mul(), and div() work similarly to add()
    
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

'''
Fill in this function to standardize each column of the given
DataFrame. To standardize a variable, convert each value to the
number of standard deviations it is above or below the mean.

This time, try to use vectorized operations instead of apply().
You should get the same results as you did before.
'''
    
def standardize(df):
    mean = df.mean(axis = 0)
    std = df.std(axis = 0,ddof = 0)
    st = df.sub(mean, axis='columns')
    return st.div(std, axis='columns')

def standardize2(df):      # Alternative concise solution
    return (grades_df - grades_df.mean()) / grades_df.std()

standardize(grades_df) # Output should be same as before (line 461)


def standardize_rows(df):
    mean = df.mean(axis = 1)
    std = df.std(axis = 1,ddof = 0)
    st = df.sub(mean, axis='index')
    return st.div(std, axis='index')

def standardize_rows2(df):     # Alternative solution
    mean_diffs = grades_df.sub(grades_df.mean(axis='columns'), axis='index')
    return mean_diffs.div(grades_df.std(axis='columns'),axis='index')

standardize_rows2(grades_df) # Weird output? -> only 2 values per row?

# Note: In order to get the proper computations, we should actually be
# setting the value of the "ddof" parameter to 0 in the .std() function.
# Note that the type of standard deviation calculated by default is different
# between numpy's .std() and pandas' .std() functions. By default, numpy calculates
# a population standard deviation, with "ddof = 0". On the other hand, pandas
# calculates a sample standard deviation, with "ddof = 1". If we know all of the scores,
# then we have a population - so to standardize using pandas, we need to set "ddof = 0".

#####################################
#       11: Pandas groupby()        #
#####################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for each block of code to see what it does

# Examine DataFrame
if True:
    print(example_df)
    
# Examine groups
if True:
    grouped_data = example_df.groupby('even')
    # The groups attribute is a dictionary mapping keys to lists of row indexes
    print(grouped_data.groups)
    
# Group by multiple columns
if True:
    grouped_data = example_df.groupby(['even', 'above_three'])
    print(grouped_data.groups)
    
# Get sum of each group
if True:
    grouped_data = example_df.groupby('even')
    print(grouped_data.sum())
    
# Limit columns in result
if True:
    grouped_data = example_df.groupby('even')
    
    # You can take one or more columns from the result DataFrame
    print(grouped_data.sum()['value'])
    
    print('\n')# Blank line to separate results
    
    # You can also take a subset of columns from the grouped data before 
    # collapsing to a DataFrame. In this case, the result is the same.
    print(grouped_data['value'].sum())
    
filename = 'nyc_subway_weather.csv'
subway_df = pd.read_csv(filename)
subway_df.head()
ridership_by_day = subway_df.groupby('day_week').mean()['ENTRIESn_hourly']

ridership_by_day.plot()

### Write code here to group the subway data by a variable of your choice, then
### either print out the mean ridership within each group or create a plot.
    # categorical variables: rain, weekday, fog, station (n = 207 though)

#####################################
#   12: Calculating houlry data     #
#####################################

# To clarify the structure of the data, the original data recorded the cumulative
# number of entries on each station at four-hour intervals. For the quiz, you just
# need to look at the differences between consecutive measurements on each station:
# by computing "hourly entries", we just mean recording the number of new
# tallies between each recording period as a contrast to "cumulative entries".

import numpy as np
import pandas as pd

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for each block of code to see what it does

# Standardize each group
if True:
    def standardize(xs):
        return (xs - xs.mean()) / xs.std()
    grouped_data = example_df.groupby('even')
    print(grouped_data['value'].apply(standardize))
    
# Find second largest value in each group
if True:
    def second_largest(xs):
        sorted_xs = xs.sort_values(inplace=False, ascending=False)
        return sorted_xs.iloc[1]
    grouped_data = example_df.groupby('even')
    print(grouped_data['value'].apply(second_largest))

# --- Quiz ---
# DataFrame with cumulative entries and exits for multiple stations
ridership_df = pd.DataFrame({
    'UNIT': ['R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051'],
    'TIMEn': ['00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00', '12:00:00', '14:00:00', '16:00:00'],
    'ENTRIESn': [3144312, 8936644, 3144335, 8936658, 3144353, 8936687, 3144424, 8936819, 3144594],
    'EXITSn': [1088151, 13755385,  1088159, 13755393,  1088177, 13755598, 1088231, 13756191,  1088275]
})

'''
Fill in this function to take a DataFrame with cumulative entries
and exits and return a DataFrame with hourly entries and exits.
The hourly entries and exits should be calculated separately for
each station (the 'UNIT' column).

Hint: Take a look at the `get_hourly_entries_and_exits()` function
you wrote in a previous quiz, DataFrame Vectorized Operations. If
you copy it here and rename it, you can use it and the `.apply()`
function to help solve this problem.
'''
    
def hourly_for_group(entries_and_exits):
    return entries_and_exits - entries_and_exits.shift()

ridership_df.groupby('UNIT').apply(hourly_for_group) # doesn't work becazse function cannot be applied to all columns
ridership_df.groupby('UNIT')[['ENTRIESn','EXITSn']].apply(hourly_for_group)


#####################################
#  13: Multiple dataframes, merge   #
#####################################

import pandas as pd

subway_df = pd.DataFrame({
    'UNIT': ['R003', 'R003', 'R003', 'R003', 'R003', 'R004', 'R004', 'R004',
             'R004', 'R004'],
    'DATEn': ['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
              '05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ENTRIESn': [ 4388333,  4388348,  4389885,  4391507,  4393043, 14656120,
                 14656174, 14660126, 14664247, 14668301],
    'EXITSn': [ 2911002,  2911036,  2912127,  2913223,  2914284, 14451774,
               14451851, 14454734, 14457780, 14460818],
    'latitude': [ 40.689945,  40.689945,  40.689945,  40.689945,  40.689945,
                  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ],
    'longitude': [-73.872564, -73.872564, -73.872564, -73.872564, -73.872564,
                  -73.867135, -73.867135, -73.867135, -73.867135, -73.867135]
})

weather_df = pd.DataFrame({
    'DATEn': ['05-01-11', '05-01-11', '05-02-11', '05-02-11', '05-03-11',
              '05-03-11', '05-04-11', '05-04-11', '05-05-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'latitude': [ 40.689945,  40.69132 ,  40.689945,  40.69132 ,  40.689945,
                  40.69132 ,  40.689945,  40.69132 ,  40.689945,  40.69132 ],
    'longitude': [-73.872564, -73.867135, -73.872564, -73.867135, -73.872564,
                  -73.867135, -73.872564, -73.867135, -73.872564, -73.867135],
    'pressurei': [ 30.24,  30.24,  30.32,  30.32,  30.14,  30.14,  29.98,  29.98,
                   30.01,  30.01],
    'fog': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'rain': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'tempi': [ 52. ,  52. ,  48.9,  48.9,  54. ,  54. ,  57.2,  57.2,  48.9,  48.9],
    'wspdi': [  8.1,   8.1,   6.9,   6.9,   3.5,   3.5,  15. ,  15. ,  15. ,  15. ]
})

'''
Fill in this function to take 2 DataFrames, one with subway data and one with weather data,
and return a single dataframe with one row for each date, hour, and location. Only include
times and locations that have both subway data and weather data available.
'''

def combine_dfs(subway_df, weather_df):
    return subway_df.merge(weather_df,on=['DATEn','hour','latitude','longitude'],how='inner')
combine_dfs(subway_df, weather_df)

### merge()
    # 'on' determines what variable the DFs are matched on
        # can be multiple
        # 'left_on' and 'right_on' to match columns that hve different names on each side
    # 'how' is important
        # inner -> only rows with 'on' value present in both DFs are kept 
        # right -> rows from 'right-hand table' (the one in parentheses aafter merge)
        #          are kept even if they're not present in the other DF (NaN)
        # left -> opposite of that
        # outer -> all rows from both DFs are kept, NaN's are filled if needed    

#####################################
#    14: Plotting for dataframes    #
#####################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for this block of code to see what it does

# groupby() without as_index
if True:
    first_even = example_df.groupby('even').first()
    print(first_even)
    print(first_even['even']) # Causes an error. 'even' is no longer a column in the DataFrame
    
# groupby() with as_index=False
if True:
    first_even = example_df.groupby('even', as_index=False).first()
    print(first_even)
    print(first_even['even']) # Now 'even' is still a column in the DataFrame

filename = 'nyc_subway_weather.csv'
subway_df = pd.read_csv(filename)

# scatterplot of stations with latitude & longitude
data_by_location = subway_df.groupby(['latitude','longitude'],as_index=False).mean()
data_by_location.head()['latitude'].head()

scaled_entries = (data_by_location['ENTRIESn_hourly'] /
                  data_by_location['ENTRIESn_hourly'].std())

plt.scatter(data_by_location['latitude'],data_by_location['longitude'],
            s=scaled_entries)



## Make a plot of your choice here showing something interesting about the subway data.
## https://matplotlib.org/2.0.2/api/pyplot_api.html
## Once you've got something you're happy with, share it on the forums!