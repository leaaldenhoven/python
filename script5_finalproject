##################################
#          Final Project         #
##################################

# Brainstorm some questions you could answer using the data set you chose, then start answering those questions. Here are some ideas to get you started:

# Titanic Data
# What factors made people more likely to survive?
# Baseball Data
# What is the relationship between different performance metrics? Do any have a strong negative or positive relationship?
# What are the characteristics of baseball players with the highest salaries?
# Make sure you use NumPy and Pandas where they are appropriate!

import pandas as pd

filename = 'titanic_data.csv'
titanic_df = pd.read_csv(filename)
titanic_df.head()
titanic_df.columns
    # Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
    #       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
    
    # Pclass = ticket class (1st, 2nd, 3rd) proxy for SES
    # sibsp = nr of siblings/spouses aboard
    # parch = nr of parents/children aboard
    # embarked = port of embarkation (Cherbourg, Queenstown, Southampton)


### Q1: How many people were on the Titanic? ###
len(titanic_df)

### Q2: How many of those people survived? ###
titanic_df['Survived'].sum()
titanic_df['Survived'].sum()/len(titanic_df)

### How many were adults/chidlren? ###
adults_df = titanic_df.loc[titanic_df['Age'] >= 21]
len(adults_df)
len(titanic_df) - len(adults_df)

### Q3: What was the gender/age balance? "Women and children first" ###
    # How many women/men were in the adult population?
adults_df.groupby(['Sex']).count()
    # How many adult men/women survived?
adults_df.groupby(['Sex','Survived'])['PassengerId'].count() 

### Q4: How old were passengers? ###
titanic_df['Age'].describe()

### Q5: How old were the surviving passengers? ###
titanic_df.groupby(['Survived'])['Age'].describe()

### Q6: 



    # Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
    #       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
    
    # Pclass = ticket class (1st, 2nd, 3rd) proxy for SES
    # sibsp = nr of siblings/spouses aboard
    # parch = nr of parents/children aboard
    # embarked = port of embarkation (Cherbourg, Queenstown, Southampton)
