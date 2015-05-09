# -*- coding: utf-8 -*-
"""
Created on Thu May 07 15:57:21 2015

@author: Lingwei
"""
##import package
import numpy as np
import pandas as pd
import statsmodels.api as sm
from bokeh.plotting import figure, output_file, show


##import data, create variables 
data = pd.read_csv('C:/Users/Lingwei/Documents/ConnectCubed_test/AirPassengers.csv', index_col=0)
X=data.time
y=data.AirPassengers

##run regression and output results
XX = sm.add_constant(X)
est = sm.OLS(y, XX)
result = est.fit()
print (result.summary())

## plot the data and best fit line
y_hat=result.predict()
output_file("plot.html", title="Airpassengers")
p = figure(title="Air Passengers", x_axis_label='year', y_axis_label='Number of Passengers')
p.scatter(X, y, size=5, color="red")
p.line(X, y_hat, legend="Best Fit Line",color='blue', line_width=2)
show(p)
