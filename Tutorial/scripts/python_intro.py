#%%
###############################################################################
# The basics
###############################################################################

# Print function

print("Today is the first tutorial of the machine learning course.")

#%%
# Variable types
"""
The two most important types of numbers in Python are:
    1. Integers: whole numbers
    2. Floats: decimals
"""

my_integer = 7
print(my_integer)
print(type(my_integer))

my_float = 7.0
print(my_float)
print(type(my_float))

#%%
# Strings
"""
Strings are a sequence of unicode characters. They can be initiated by either +
a single quote oder double quote.
"""

my_string = "Hello"
print(my_string)

my_string = 'Hello'
print(my_string)
print(type(my_string))

# String operatings

hello = "hello"
world = "world"

hello_world = hello + " " + world
print(hello_world)#

a = "AskPython"
print(a[2:8]) # A colon used on the right side of the index will display the 
              # everything after that particular index as an output. This will 
              # not display the index that is mentioned in the code.
print(a[3:])
print(a[:4])

#%%
# Dictionaries

"""
- Dictionaries store items in key-value pairs.
- A dictionary is a collection of items, which is ordered, changeable but does not
allow duplicates.      
"""
my_dict = {
    "student_first_name": "Peter",
    "student_last_name": "Mayer",
    "age": 28
    }

print(my_dict)
print(type(my_dict))

print(my_dict["age"])

# Dictionaries can not have two items with the same key. Duplicated values will
# overwrite existing values
my_dict = {
    "student_first_name": "Peter",
    "student_last_name": "Mayer",
    "age": 28,
    "age" : 23
    }

print(my_dict)


my_dict_phonebook = {
    "Peter" : {
        "phone": 1234,
        "mail": "peter_mayer@uni_hamburg.de"
        },
    "Hanna":{
        "phone": 5678,
        "mail": "hanna_müller@uni_hamburg.de"
        }
    }

print(my_dict_phonebook)
print(my_dict_phonebook["Peter"])
print(my_dict_phonebook["Peter"]["phone"])
#%%
# Tuple
"""
- Tuple is a collection of items, which is ordered and unchangeable.
- A tuple allows for duplicated items
"""

my_tuple = (1,3,5,5, "hello")
print(my_tuple)
print(my_tuple[0])
print(type(my_tuple))
#%%
# Lists
"""
- Lists can contain any type of varibale and can contain any amount of variables. 
Furthermore, lists are iterable.
"""
# Initiate a list
my_list = []

# Append variables to a lists
my_list.append(1)
my_list.append(2)
my_list.append("hello")

print(my_list)
print(type(my_list))

# Iterate over a lists
for item in my_list:
    print(item)
    
# iterate over a list with enumerate
for idx, item in enumerate(my_list):
    print(idx, item)

print(my_list[0])
print(my_list[3])


#%%
# Arithmetic Operations 

"""
Use usual arithemtic operations with numbers
"""

# Standard calculations
calculation = 2 + 5 *4 -3 / 4
print(calculation)

# Arithmetic operations with lists
even_numbers = [2,4,6,8]
odd_numbers = [1,3,5,7]

# Join two lists
joined = even_numbers + odd_numbers
print(joined)

# Cycle a list
print(even_numbers*3)



#%%
# String Formatting

"""
There are several possibilities to format strings in Python. Python uses C-style
string formatting to create new, formatted strings. The "%" operator is used to
format a set of variables enclosed in a "tuple", together with a format string,
which contains normal text together with "argument specifiers", special symbols
like "%s" or "%d".

    - %s: string (or any object with a string representation.)
    - %d: integers
    - %f. floats
"""

name = "John"
age = 22

print("Hi my name is %s and I am %d years old" % (name, age))

#%%
# Conditions

"""
Python uses boolean logic to evaluate conditions. The boolean values "True" and
"False" are returned when an expression is compared or evaluated.
"""

x = 2
print(x==2)
print(x==3)
print(x<3)

# The "and" and "or" boolean operators allow building complex expressions. 
name = "Peter"
age = 23

if name == "Peter" and age == 23.:
    print("Your name is %s and you are also %d years old." % (name, age))
    
if name == "Peter" or name == "Hanna":
    print("Your name is either Peter or Hanna")


# The "in" operator is used to check if a specific object exists within an 
# iterable object container, such as a list.

name = "Peter"
if name in ["Peter", "Hanna"]:
    print("your name is either Peter or Hanna")
    

# Unlike the "==" operator, the "is" operator does not match the values of the 
# variables, but the instances themselves.

x = [1,2,3]
y = [1,2,3]

print(x == y)
print(x is y)

# The "not" operator inverts the expression

print(x is not y)

#%%
###############################################################################
# Loops 
###############################################################################

# For-Loops

"""
For-Loops iterate over a given sequence.
"""

numbers = [1,2,3,4,5,6]

for number in numbers:
    print(number)
    
# For-Loops can iterate over a sequence of numbers using the "range"-command. 
# Note, that range starts counting at zero and end at the penultimate number.

for x in range(5):
    print(x)
    
for x in range(3,8):
    print(x)
    


# While-Loops
"""
While-Loops repeat as long as a certain boolean is True.
"""
 
count = 0
while count < 5:
    print(count)
    count += 1

#%%
# Break and continue statements

"""
Break is used to exit a for-loop or while-loop, whereas continue is used to skip 
the current block and return to the "for" or "while" statement.
"""     

count = 0

while True:
    print(count)
    count += 1
    if count >= 5:
        break
    
for x in range(10):
    if x % 2 == 0:
        continue
    print(x)
    
lookatrange=range(5) 
     
#%%
# Else clause in loops

"""
When the loop condition "for" or "while" fails, then the code part "else" is executed.
If a break statement is executed inside the for-loop then the "else" part is skipped.
Note, that the "else" part is executed even if there is a continue statement.
"""

count = 0
while(count < 5):
    print(count)
    count += 1
else:
    print("count value reached %d" %(count))
    
#%%
###############################################################################
# Functions
###############################################################################

"""
Functions are a good way to divide code into useful chunks, allowing us to ordere
the code, make it more readable, reuse it and save some time.
"""

# Write functions
def my_function():
    print("Hello from my_function")
    
# functions may also receive arguments. Arguments are variables passed from the 
# caller to the function.

def my_function_with_args(username, greeting):
    print("Hello, %s from my_function! I wish you %s" %(username, greeting))
    

# functions may also return a value to the caller. This is done by using the 
# keyword "return".

def sum_two_numbers(a,b):
    return a + b

#%%
# Call Functions
my_function()

my_function_with_args("Peter", "a great day")

x = sum_two_numbers(1, 2)
print(x)

#%%
###############################################################################
# Pandas 
###############################################################################

"""
Pandas is a high-level data manipulation tool developed by Wes McKinney. 
Its key data structure is called the DataFrame. 
DataFrames allow you to store and manipulate tabular data in rows of observations 
and columns of variables. There are several ways to create a DataFrame.
One way way is to use a dictionary.
"""

dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

import pandas as pd
brics = pd.DataFrame(dict)
print(brics)


# If you would like to have different index values, say, the two letter country
# code. 
# Set the index for brics
brics.index = ["BR", "RU", "IN", "CH", "SA"]

# Print out brics with new index values
print(brics)


# Another way to create a Dataframe is by importing a file using Pandas
df = pd.read_csv("Desktop/machine_learning_tutorial/titanic.csv")

#%%
# Explorative Data Analysis (EDA)

print(df.shape)
print("."*100)
print(df.columns)
print("."*100)
print(df.head(10))
print("."*100)
print(df.isnull().sum())
print("."*100)
print(df.describe())
print("."*100)
print(df.dtypes)


#%%
# Indexing Data
# Use square brackets for indexing
print(df["Sex"])
print("."*100)
print(df[["Sex", "Survived"]])
print("."*100)
# Accessros -> loc & iloc
print(df.loc[:500,"Sex"]) # loc function selects columns/rows using row labels 
print("."*100)
print(df.loc[500:,"Sex"])
print("."*100)
print(df.iloc[:500, 5]) #iloc function selects columns/rows using integer positions
print("."*100)
print(df.iloc[:, 5])
print("."*100)
print(df.iloc[:, 2:5])

# Slicing specific values from dataframe
print(df["Name"][1:4]) 
print(df["Name"][2])

#%%
# Filter data
female_series = df["Sex"] == "female"
female_dataframe = df[df["Sex"] == "female"]

older_35 = df[df["Age"] > 35.]
male_and_older_35 = df[(df["Sex"] == "male") & (df["Age"] > 35.)]

#%% 
# Create new columns

df["younger_than_35"] = 0
df["younger_than_35"] = [1 if x < 35 else 0 for x in df["Age"]]

#%%
# Clean data

nan_values = df.isna().sum()
print(nan_values)

# drop nan values
df_without_nan_axis0 = df.dropna(axis = 0) # axis=0 for rows
print(df_without_nan_axis0.isna().sum())

df_without_nan_axis1 = df.dropna(axis = 1) # axis=1 for columns
print(df_without_nan_axis1.isna().sum())

#%%
# Working with Dataframes
df.sort_values(by = "Survived", inplace = True)
df = df.sort_values(by = "Survived")

grouped_by_sex = df.groupby("Sex").count()
print(grouped_by_sex)

print(df.groupby(["Sex", "Survived"])["Age"].mean())

#%%
# Plotting data

import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')

sb.countplot(df, x="Survived")

df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
sb.countplot(x='Sex',hue='Survived',data=df)
plt.show()

pd.crosstab([df.Sex,df.Survived],df.Pclass,margins=True)



















































