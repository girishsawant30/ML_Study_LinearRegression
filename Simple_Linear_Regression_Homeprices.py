import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("E:\Girish Documents\Study\Data Science\DataScience_ML_Study\ML_Study_LinearRegression\Simple_Linear_Regression_Homeprices.csv")
#print(df.head())

plt.scatter(df.area, df.price, color='red', marker='+')
plt.xlabel("Area(sqr ft)")
plt.ylabel("Price(USD)")
#plt.show()

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(df[['area']],df.price)


'''This code is to pickup the file for the Single Area 3000'''
print(reg.predict([[3300]]))
c = reg.coef_
i = reg.intercept_
print("Coef & Intercept are :{} & {}".format(c,i) )
print(c*3300+i)

'''This code is to pickup the file for the multiple Areas'''
d = pd.read_csv("E:\Girish Documents\Study\Data Science\DataScience_ML_Study\ML_Study_LinearRegression\Simple_Linear_Regression_Homeprices_TobePredicted_AreasList.csv")
#print(d.head())
#print(reg.predict(d))


'''This code is to export the predicted areas in an excel'''
d['prices'] = reg.predict(d)
#print(d)
d.to_csv("E:\Girish Documents\Study\Data Science\DataScience_ML_Study\ML_Study_LinearRegression\Simple_Linear_Regression_Homeprices_TobePredicted_AreasList_Predicted.csv", index=False)

'''This code is for plotting the line on the graph'''
plt.xlabel('area', fontsize=10)
plt.ylabel('price', fontsize=10)
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()
