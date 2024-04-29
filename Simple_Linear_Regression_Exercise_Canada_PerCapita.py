import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("E:\Girish Documents\Study\Data Science\DataScience_Study\LinearRegression_1\Simple_Linear_Regression_exercise_canada_percapita.csv")
print(df.head())
df.rename(columns={"per capita income (US$)":"PerCapita"},inplace=True )
print(df.head())


plt.xlabel("Year")
plt.ylabel("Income PerCapita")
plt.scatter(df.year,df.PerCapita, marker='+',color='red')
#plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df[['year']],df.PerCapita)

#reg = linear_model.LinearRegression()
#reg.fit(df[['area']],df.price)
print(reg.predict([[2020]]))


plt.xlabel("Year")
plt.ylabel("Income PerCapita")
plt.scatter(df.year,df.PerCapita, marker='+',color='red')
plt.plot(df.year, reg.predict(df[['year']]), color='blue')

plt.show()
