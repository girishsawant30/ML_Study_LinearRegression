import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("E:\Girish Documents\Study\Data Science\DataScience_Study\Simple_Linear_Regression_Salary_dataset.csv")
#print(df.head())
selected_columns = df[['YearsExperience', 'Salary']]
#print(selected_columns.head())
#print(selected_columns.describe())

plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(df.YearsExperience, df.Salary, marker='+', color='blue')
#plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(selected_columns[['YearsExperience']].values, selected_columns.Salary.values)
pred_value = reg.predict([[1.5]])
print(pred_value)

m = reg.coef_
c = reg.intercept_
print(m*1.5+c)

plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(selected_columns.YearsExperience, selected_columns.Salary, marker='+', color='blue')
plt.plot(selected_columns.YearsExperience, reg.predict(selected_columns[['YearsExperience']].values), color='red')
plt.show()