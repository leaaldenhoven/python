#################################################
#              Numpy and Pandas                 #
#################################################

#################################################
#           1: Generating Questions             #
#################################################

### Gapminder Dataset
# employment levels - life expectancy - GDP - school completion rates


# Q1: Has employment changed over time
# Q2: Does employment differ between countries
# Q3: Do different countries have different trajectories of employment-change over time
# Q4: same questions for GDP --> important, here is some missing data
# Q5: same questions for life expectancy --> differences in trajectories especially interesting here
    # again issue of missing data
# Q6: same questions for school completion
# Q7: Comparison between male/female in terms of school completion (interaction with country, role of equal opporunities)
# Q8: metadata: compare from when records are available (less interesting?)
# Q9: relationships between variables
    # when comparing different variables -> note differences in missing data & timepoints!
    # since we have dates, excellent opportunity to look at school completion of particular cohort
    # and consequent employment of estimated similar cohort
    
# More focused questions/suggestions from teacher -> focus on one country: Canada/Germany, here US
# Q1: How has employment in the US varied over time?
# Q2: What are the highest/lowest employment levels?
    # Where does the US/Canada/Germany stand on that spectrum
# Q3-5: same questions for other variables
# Q6: How do the variables relate
# Q7: Are there consistencies across countries? e.g. global recessions, ...


#################################################
#    2: Read In Data - why pandas is better     #
#################################################

### Previous code -> 14 lines, takes 2 minutes, used up to 90% of Mem

import unicodecsv #library
def read_csv(filename):
    with open(filename,'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)
daily_engagement = read_csv('Lesson 1  Data Analyst Process/daily_engagement_full.csv')

def get_unique_students(data):
    unique_students = set()
    for data_point in data:
        unique_students.add(data_point['account_key'])
    return unique_students
engagement_num_unique_students = get_unique_students(daily_engagement)
len(engagement_num_unique_students)

### With pandas -> 3 lines, takes 8 seconds, used up to 60% Mem

import pandas as pd
daily_engagement = pd.read_csv('Lesson 1  Data Analyst Process/daily_engagement_full.csv')
len(daily_engagement['acct'].unique())


#################################################
#           3: 1D data structures               #
#################################################

### Pandas - series
    # more features

### NumPy - array ###
    # simpler
    # basis of series
    # -> so good to learn first
    
### NumPy arrays are similar to lists
       # a = ['A','B','C','D']
    # Access elements by position:  a[0] -> 'A'
    # Access a range of elements:   a[1:3] -> 'B','C','D'
    # Use loops:                    for x in a: ...
    
### NumPy arrays are different to lists
    # each element should have same type!
    # arrays have convenient functions -> mean(), std()
    # can be multidimensional (similar to list of lists)

#################################################
#       4: Playing around with arrays           #
#################################################

### Using only a small section of the data

import numpy as np

# First 20 countries with employment data
countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])

# Employment data in 2007 for those 20 countries
employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])

# Change False to True for each block of code to see what it does

# Accessing elements
if True:
    print(countries[0])
    print(countries[3])

# Slicing
if True:
    print(countries[0:3])
    print(countries[:3])
    print(countries[17:])
    print(countries[:])

# Element types
if True:
    print(countries.dtype)
    print(employment.dtype)
    print(np.array([0, 1, 2, 3]).dtype)
    print(np.array([1.0, 1.5, 2.0, 2.5]).dtype)
    print(np.array([True, False, True]).dtype)
    print(np.array(['AL', 'AK', 'AZ', 'AR', 'CA']).dtype)

# Looping
if True:
    for country in countries:
        print('Examining country {}'.format(country))

    for i in range(len(countries)):
        country = countries[i]
        country_employment = employment[i]
        print('Country {} has employment {}'.format(country,
                country_employment))

# Numpy functions
if True:
    print(employment.mean())
    print(employment.std())
    print(employment.max())
    print(employment.sum())

# Finding the maximum

def max_employment(countries, employment):
    max_value = 0
    for i in range(len(employment)):
        if employment[i] > max_value:
            max_value = employment[i]
            max_country = countries[i]
    return (max_country, max_value)

max_employment(countries, employment) #('Angola', 75.69999695)

# An easier way using NumPy
def max_employment2(countries, employment):
    j = employment.argmax()
    return (countries[j],employment[j])
max_employment2(countries, employment) #('Angola', 75.69999695)

#################################################
#           5: Vectorized Operations            #
#################################################

# vector = list of numbers
# vectors can be added & multipled by a scalar
   # depending on programming language
        # (1 2 3) + (4 5 6) = ?
        # (1 2 3) * 3 = ?
    # numpy adds array following rules of linear algebra
        # each element of the vector is added to the
        # corresponding element in the other vector
        # = (5 7 9)
        # = (3 6 9)
    # list concatenation in python
        # = (1 2 3 4 5 6)
        # = (1 2 3 1 2 3 1 2 3)
    # others
        # err, very common

# Other operators that work with arrays/vectors -> cheatsheet
    # Math operations
        #  + - * / **
    # Logical operations (for boolean arrays)
        #  & | ~
    # Comparison operations
        #  > >= < <= == !=
        
### Experiment with operators & vectors

import numpy as np

# Change False to True for each block of code to see what it does

# Arithmetic operations between 2 NumPy arrays
if False:
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 1, 2])
    
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(a ** b)
    
# Arithmetic operations between a NumPy array and a single number
if False:
    a = np.array([1, 2, 3, 4])
    b = 2
    
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(a ** b)
    
# Logical operations with NumPy arrays
if False:
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    
    print(a & b)
    print(a | b)
    print(~a)
    
    print(a & True)
    print(a & False)
    
    print(a | True)
    print(a | False)
    
# Comparison operations between 2 NumPy Arrays
if False:
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1])
    
    print(a > b)
    print(a >= b)
    print(a < b)
    print(a <= b)
    print(a == b)
    print(a != b)
    
# Comparison operations between a NumPy array and a single number
if False:
    a = np.array([1, 2, 3, 4])
    b = 2
    
    print(a > b)
    print(a >= b)
    print(a < b)
    print(a <= b)
    print(a == b)
    print(a != b)


### Calculate overall completion rate, assuming 1:1 male:female distribution
    
# First 20 countries with school completion data
countries = np.array([
       'Algeria', 'Argentina', 'Armenia', 'Aruba', 'Austria','Azerbaijan',
       'Bahamas', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Bolivia',
       'Botswana', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi',
       'Cambodia', 'Cameroon', 'Cape Verde'
])

# Female school completion rate in 2007 for those 20 countries
female_completion = np.array([
    97.35583,  104.62379,  103.02998,   95.14321,  103.69019,
    98.49185,  100.88828,   95.43974,   92.11484,   91.54804,
    95.98029,   98.22902,   96.12179,  119.28105,   97.84627,
    29.07386,   38.41644,   90.70509,   51.7478 ,   95.45072
])

# Male school completion rate in 2007 for those 20 countries
male_completion = np.array([
     95.47622,  100.66476,   99.7926 ,   91.48936,  103.22096,
     97.80458,  103.81398,   88.11736,   93.55611,   87.76347,
    102.45714,   98.73953,   92.22388,  115.3892 ,   98.70502,
     37.00692,   45.39401,   91.22084,   62.42028,   90.66958
])

def overall_completion_rate(female_completion, male_completion):
    return (female_completion + male_completion)/2

overall_completion_rate(female_completion, male_completion)

#################################################
#           6: Standardizing Values             #
#################################################

import numpy as np

# First 20 countries with employment data
countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])

# Employment data in 2007 for those 20 countries
employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])

# Change this country name to change what country will be printed when you
# click "Test Run". Your function will be called to determine the standardized
# score for this country for each of the given 5 Gapminder variables in 2007.
# The possible country names are available in the Downloadables section.

country_name = 'United States'

def standardize_data(values):
    standardized_values = (values - values.mean())/values.std()
    return standardized_values

#################################################
#                 7: Index Arrays               #
#################################################

a = np.array([1,2,3,4,5])
b = np.array([False, False, True, True, True])
a[b] # only keep the indices of array a that are true in b
b = a > 2 # simpler version
a[a>2] # simpler version (b not needed)

# Play around with index arrays

import numpy as np

# Change False to True for each block of code to see what it does

# Using index arrays
if False:
    a = np.array([1, 2, 3, 4])
    b = np.array([True, True, False, False])
    
    print(a[b])
    print(a[np.array([True, False, True, False])])
    
# Creating the index array using vectorized operations
if False:
    a = np.array([1, 2, 3, 2, 1])
    b = (a >= 2)
    
    print(a[b])
    print(a[a >= 2])
    
# Creating the index array using vectorized operations on another array
if False:
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 2, 1])
    
    print(b == 2)
    print(a[b == 2])

# Time spent in the classroom in the first week for 20 students
time_spent = np.array([
       12.89697233,    0.        ,   64.55043217,    0.        ,
       24.2315615 ,   39.991625  ,    0.        ,    0.        ,
      147.20683783,    0.        ,    0.        ,    0.        ,
       45.18261617,  157.60454283,  133.2434615 ,   52.85000767,
        0.        ,   54.9204785 ,   26.78142417,    0.
])

# Days to cancel for 20 students
days_to_cancel = np.array([
      4,   5,  37,   3,  12,   4,  35,  38,   5,  37,   3,   3,  68,
     38,  98,   2, 249,   2, 127,  35
])
    
def mean_time_for_paid_students(time_spent, days_to_cancel):
    return time_spent[days_to_cancel >= 7].mean()

# longer version
def mean_time_for_paid_students2(time_spent, days_to_cancel):
    is_paid = days_to_cancel >= 7
    paid_time = time_spent[is_paid]
    return paid_time.mean()
mean_time_for_paid_students(time_spent, days_to_cancel)

#################################################
#        8: += vs +  in-place vs. not           #
#################################################

import numpy as np
a = np.array([1,2,3,4])
b = a
a += np.array[(1,1,1,1)]
print(b)
# Output: [(2,3,4,5)]
# both a and b will be updated ->  += operates "in-place"

import numpy as np
a = np.array([1,2,3,4])
b = a
a = a + np.array[(1,1,1,1)]
print(b)
# Output: [(1,2,3,4)]
# ONLY a will be updated ->  + operates NOT "in-place"
# using operations NOT in-place can be easier to think about

import numpy as np
a = np.array[(1,2,3,4,5)]
slice = a[:3]
slice[0] = 100
print(a)
# Output: [(100,2,3,4,5)]
# slice is a view of the array, not a new array!
# when we update the slice, we also update the array! be careful!


#################################################
#                9: Pandas Series               #
#################################################

# like array but with more functionalities  e.g. s.describe()
# Similarities: all the previous functions work for series

### Playing around with pandas
# Change False to True for each block of code to see what it does

import pandas as pd

countries = ['Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
             'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
             'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
             'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia']

life_expectancy_values = [74.7,  75. ,  83.4,  57.6,  74.6,  75.4,  72.3,  81.5,  80.2,
                          70.3,  72.1,  76.4,  68.1,  75.2,  69.8,  79.4,  70.8,  62.7,
                          67.3,  70.6]

gdp_values = [ 1681.61390973,   2155.48523109,  21495.80508273,    562.98768478,
              13495.1274663 ,   9388.68852258,   1424.19056199,  24765.54890176,
              27036.48733192,   1945.63754911,  21721.61840978,  13373.21993972,
                483.97086804,   9783.98417323,   2253.46411147,  25034.66692293,
               3680.91642923,    366.04496652,   1175.92638695,   1132.21387981]

# Life expectancy and gdp data in 2007 for 20 countries
life_expectancy = pd.Series(life_expectancy_values)
gdp = pd.Series(gdp_values)

# Accessing elements and slicing
if False:
    print(life_expectancy[0])
    print(gdp[3:6])
    
# Looping
if False:
    for country_life_expectancy in life_expectancy:
        print('Examining life expectancy {}'.format(country_life_expectancy))
        
# Pandas functions
if False:
    print(life_expectancy.mean())
    print(life_expectancy.std())
    print(gdp.max())
    print(gdp.sum())

# Vectorized operations and index arrays
if False:
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([1, 2, 1, 2])
  
    print(a + b)
    print(a * 2)
    print(a >= 3)
    print(a[a >= 3])
   
def variable_correlation(variable1, variable2):
    both_above = (variable1 > variable1.mean()) & (variable2 > variable2.mean())
    both_below = (variable1 < variable1.mean()) & (variable2 < variable2.mean())
    is_same_direction = both_above | both_below
    num_same_direction = is_same_direction.sum()
    num_different_direction = len(variable1) - num_same_direction #clever
    return (num_same_direction, num_different_direction)

variable_correlation(life_expectancy, gdp) #(17,3) (same direction, diff direction)
# there is a positive correlation between life_expectancy & gdp!

#################################################
#               10: Series Indexes              #
#################################################

### Differences between numpy arrays & panda series

import numpy as np
import pandas as pd
a = np.array([1,2,3,4])
s = pd.Series([1,2,3,4])

s.describe() # Describe function

# Indexes!
numbers = pd.Series([1,2,3,4],index=['first','second','third','fourth'])
numbers

#or

a = [1,2,3,4]
b = ['first','second','third','fourth']
numbers = pd.Series(a,index=b)
numbers

# NumPy array are like souped-up Python lists
# Pandas series are like a cross between a list and a dictionary
numbers[0]
numbers.loc['second']
numbers.iloc['second'] #acces by position (avoid term index due to potential confusion)

### Practicing with indexes

countries = [
    'Afghanistan', 'Albania', 'Algeria', 'Angola',
    'Argentina', 'Armenia', 'Australia', 'Austria',
    'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
    'Barbados', 'Belarus', 'Belgium', 'Belize',
    'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
]


employment_values = [
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076,
]

# Employment data in 2007 for 20 countries
employment = pd.Series(employment_values, index=countries)

def max_employment3(employment):
    max_country = employment.idxmax()
    max_value = employment[max_country]
    return (max_country, max_value)

max_employment(employment)

def max_employment4(employment):
    max_country = employment.idxmax()
    max_value = employment.loc[max_country]
    return (max_country, max_value)

max_employment(employment)

#################################################
#     11: Vectorized Operations & Indexes       #
#################################################

import pandas as pd

# Change False to True for each block of code to see what it does

# Addition when indexes are the same
if True:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
    print(s1 + s2)

# Indexes have same elements in a different order
#-> addition happens based on index meaning not position
if True:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['b', 'd', 'a', 'c'])
    print(s1 + s2)

# Indexes overlap, but do not have exactly the same elements
# only those that match are added, AND  the rest return NaN! are not just kept as original
if True:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['c', 'd', 'e', 'f'])
    print(s1 + s2)

# Indexes do not overlap
#  as above, returns NaN
if True:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['e', 'f', 'g', 'h'])
    print(s1 + s2)

# Bottom line: series with indices are matched up based on index, not position
    
#################################################
#               12: Missing values              #
#################################################

import pandas as pd

s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([10, 20, 30, 40], index=['c', 'd', 'e', 'f'])

sum_result = s1 + s2

sum_result.dropna()

# Try to write code that will add the 2 previous series together,
# but treating missing values from either series as 0. The result
# when printed out should be similar to the following line:
# print(pd.Series([1, 2, 13, 24, 30, 40], index=['a', 'b', 'c', 'd', 'e', 'f']))

s1.add(s2, fill_value = 0)

# Remember that Jupyter notebooks will just print out the results of the last
# expression run in a code cell as though a print expression was run.
# If you want to save the results of your operations for later,
# remember to assign the results to a variable or, for some Pandas functions
# like .dropna(), use inplace = True to modify the starting object without
# needing to reassign it.

#################################################
#       13: Non-built-in calculations           #
#################################################

# So far, we used built-in functions (e.g. mean()) and operations (e.g. +)
# What to do bexond that?
    # Treat series as a list -> loops through elements
    # use "apply()" function which takes a series & a function and returns a new series
    # because otherwise data is treated as list
    # makes code concise

import pandas as pd

# Change False to True to see what the following block of code does

# Example pandas apply() usage (although this could have been done
# without apply() using vectorized operations)
if True:
    s = pd.Series([1, 2, 3, 4, 5])
    def add_one(x):
        return x + 1
    print(s.apply(add_one))

names = pd.Series([
    'Andre Agassi',
    'Barry Bonds',
    'Christopher Columbus',
    'Daniel Defoe',
    'Emilio Estevez',
    'Fred Flintstone',
    'Greta Garbo',
    'Humbert Humbert',
    'Ivan Ilych',
    'James Joyce',
    'Keira Knightley',
    'Lois Lane',
    'Mike Myers',
    'Nick Nolte',
    'Ozzy Osbourne',
    'Pablo Picasso',
    'Quirinus Quirrell',
    'Rachael Ray',
    'Susan Sarandon',
    'Tina Turner',
    'Ugueth Urbina',
    'Vince Vaughn',
    'Woodrow Wilson',
    'Yoji Yamada',
    'Zinedine Zidane'
])

name = names.iloc[0]

def reverse_name(name):
    split_name = name.split(" ")
    first_name = split_name[0]
    last_name = split_name[1]
    return last_name + "," + first_name

def reverse_names(names):
    return names.apply(reverse_name)
reverse_names(names)

# Here I was able to combine them and the corde worked fine but for exercise
# I had to split it like this doing one name first and then applying it to
# all the names

def reverse_names2(names):
    split_name = names.split(" ")
    first_name = split_name[0]
    last_name = split_name[1]
    return last_name + "," + first_name
reverse_names(names)

# Note: The grader will execute your finished reverse_names(names) function on
# some test names Series when you submit your answer. Make sure that this
# function returns another Series with the transformed names.

# split()
# You can find documentation for Python's split() function here.

#################################################
#         14: Plotting with pandas              #
#################################################

import pandas as pd
import seaborn as sns

# The following code reads all the Gapminder data into Pandas DataFrames. You'll
# learn about DataFrames next lesson.

path = "C://Users//leano//OneDrive//Documents//Udacity_IntroDataScience//Lesson 2 NumPy and Pandas//"
employment = pd.read_csv(path + 'employment_above_15.csv', index_col='Country')
female_completion = pd.read_csv(path + 'female_completion_rate.csv', index_col='Country')
male_completion = pd.read_csv(path + 'male_completion_rate.csv', index_col='Country')
life_expectancy = pd.read_csv(path + 'life_expectancy.csv', index_col='Country')
gdp = pd.read_csv(path + 'gdp_per_capita.csv', index_col='Country')

# The following code creates a Pandas Series for each variable for the United States.
# You can change the string 'United States' to a country of your choice.

employment_us = employment.loc['United States']
female_completion_us = female_completion.loc['United States']
male_completion_us = male_completion.loc['United States']
life_expectancy_us = life_expectancy.loc['United States']
gdp_us = gdp.loc['United States']

# Uncomment the following line of code to see the available country names
print(employment.index.values)


employment_de = employment.loc['Germany']
female_completion_de = female_completion.loc['Germany']
male_completion_de = male_completion.loc['Germany']
life_expectancy_de = life_expectancy.loc['Germany']
gdp_de = gdp.loc['Germany']


# Use the Series defined above to create a plot of each variable over time for
# the country of your choice. You will only be able to display one plot at a time
# with each "Test Run".

employment_de.plot()
female_completion_de.plot()
male_completion_de.plot()
life_expectancy_de.plot()
life_expectancy_us.plot()
