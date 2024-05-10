import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

data_set = pd.read_csv('SAT_GPA_Data.csv')
data_set.head()

#Exploring dataset characteristics
print("Number of records:", len(data_set))
print("Mean SAT score:", data_set['SAT'].mean())
print("Median SAT score:", data_set['SAT'].median())
print("Mean GPA:", data_set['GPA'].mean())
print("Median GPA:", data_set['GPA'].median())

#Seperating independent and dependent variables
x=data_set['SAT']   #SAT score (independent)
y=data_set['GPA']   #Salary (dependent)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Scatter Plot
plt.scatter(x,y)                   #Plotting the points
plt.xlabel('SAT', fontsize=20)     #Defining the x-Label name
plt.ylabel('GPA', fontsize=20)     #Defining the x-Label name
plt.show()

#Histogram
plt.hist(data_set['GPA'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('GPA')
plt.ylabel('Frequency')
plt.title('Distribution of GPAs Column')
plt.show()

model=LinearRegression()
x=np.array(x).reshape(-1,1)
y=np.array(y).reshape(-1,1)
results=model.fit(x,y)

beta0=results.coef_
beta1=results.intercept_
#print(beta0)
#print(beta1)

plt.scatter(x, y)
y_new = beta0*x + beta1
plt.plot(x, y_new, 'r--', lw=1, label='regression line')

plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()