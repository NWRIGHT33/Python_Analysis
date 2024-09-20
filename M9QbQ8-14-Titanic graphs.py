'''
Data analysis and graphing of titanic dataset
'''

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Number of passengers per class split by type of person using the 'ocean' palette
sns.catplot(data=titanic, x='class', hue='who', kind='count', palette='ocean')
plt.title('Number of Passengers per Class Split by Type of Person')
plt.show()

# Survival rate of passengers per class split by type of person using the 'inferno' palette
sns.catplot(data=titanic, x='class', y='survived', hue='who', kind='bar', palette='inferno')
plt.title('Survival Rate of Passengers per Class Split by Type of Person')
plt.show()


# Create the age range categories as defined
def set_age_range(age):
    if age < 18:
        return '<18'
    elif 18 <= age < 25:
        return '18-25'
    elif 25 <= age < 34:
        return '26-34'
    elif 34 <= age < 44:
        return '35-44'
    elif 44 <= age < 54:
        return '45-54'
    elif 54 <= age < 64:
        return '55-64'
    else:
        return '65+'

# Apply the function to create a new 'age_range' column
titanic['age_range'] = titanic['age'].apply(set_age_range)

# Create a new DataFrame with class, age_range, and survived columns
age_class_survived = titanic.groupby(['class', 'age_range', 'survived']).size().reset_index(name='count')

print(age_class_survived)


# Plotting the average number of survived vs. non-survived passengers per class across age ranges
sns.barplot(data=age_class_survived, x='class', y='count', hue='survived')
plt.title('Average Number of Survived vs. Deaths per Class across Age Ranges')
plt.show()

# Average number between survived and non-survived passengers per class, split by age range
sns.catplot(data=age_class_survived, x='class', y='count', hue='survived', col='age_range', kind='bar')
plt.title('Average Number between Survived and Non-Survived per Class Split by Age Range')
plt.show()

# Passengers' survival rate per class, split by age range
sns.catplot(data=age_class_survived, x='class', y='count', hue='survived', col='age_range', kind='point')
plt.title('Survival Rate per Class Split by Age Range')
plt.show()

# Scatter plots showing dependencies 
# Scatter plot between age and fare
sns.scatterplot(data=titanic, x='age', y='fare')
plt.title('Age vs. Fare')
plt.show()

# Scatter plot between age and fare, with hue based on sex
sns.scatterplot(data=titanic, x='age', y='fare', hue='sex')
plt.title('Age vs. Fare (Split by Sex)')
plt.show()

# Scatter plot between age and fare, with hue based on who (type of person) and different size
sns.scatterplot(data=titanic, x='age', y='fare', hue='who', size='who', sizes=(40, 400))
plt.title('Age vs. Fare (Split by Type of Person with Different Sizes)')
plt.show()

# Scatter plot with different colors based on who
sns.scatterplot(data=titanic, x='age', y='fare', hue='who', palette='coolwarm')
plt.title('Age vs. Fare (Different Colors for Each Type of Person)')
plt.show()

# catter plot with different shapes for each type of person
markers = {"man": "o", "woman": "s", "child": "D"}
sns.scatterplot(data=titanic, x='age', y='fare', hue='who', style='who', markers=markers)
plt.title('Age vs. Fare (Different Shapes for Each Type of Person)')
plt.show()

# Multiple subplots (by gender) showing age vs. fare
sns.relplot(data=titanic, x='age', y='fare', hue='who', col='sex')
plt.title('Age vs. Fare (Multiple Subplots by Gender)')
plt.show()

# Joint plot of age and fare with histplot marginal distributions and linear regression
sns.jointplot(data=titanic, x='age', y='fare', kind='reg', marginal_kws={'bins': 20})
plt.title('Joint Plot of Age and Fare with Regression Line')
plt.show()


# Scatter plot with rugplot for age and fare
sns.scatterplot(data=titanic, x='age', y='fare')
sns.rugplot(data=titanic, x='age', color='blue', height=0.05)
sns.rugplot(data=titanic, y='fare', color='red', height=0.05)
plt.title('Scatter Plot of Age and Fare with Rug Plot')
plt.show()
