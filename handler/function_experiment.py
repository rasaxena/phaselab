import matplotlib.pyplot as plt
import numpy as np

def my_function(x, a1, a2,b1,b2,c):
    # print("a1",a1,"a2",a2, "b1", b1, "b2",b2, "c",c)
    return (a1**3.5) * np.exp(-1.5*b1 * x) - a2**0.5* np.exp(-2.5*b2 * x)+c*x


import pickle

popt = pickle.load(open('../save2x','rb'))
a1 = popt[0][0][0]
a2 = popt[0][0][1]
b1 = popt[0][0][2]
b2 = popt[0][0][3]
c = popt[0][0][4]

# a1= a1+0.01
b2 = b2 - 0.01
print(a1,a2,b1, b2,c)

x = np.linspace(2,300,2980)

y = my_function(x,a1,a2,b1,b2,c)
import plotly.graph_objs as go


import plotly.plotly as py
py.sign_in('rangolisaxena90', 'BqBUFqjQKYa7viDwRUHN')
from plotly.tools import FigureFactory as FF
import pandas as pd

dy = np.zeros(y.shape,np.float)
dy[0:-1] = np.diff(y)/np.diff(x)
dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
trace1 = go.Scatter(
    x=x,
    y=y,
    mode='line',
    name='pdd_function(x)'
)

trace2 = go.Scatter(
    x=x,
    y=dy,
    mode='line',
    name='numerical derivative of pdd_function(x)'
)

trace_data = [trace1, trace2]
print(trace_data)
py.plot(trace_data, filename='numerical-differentiation')


print(popt)