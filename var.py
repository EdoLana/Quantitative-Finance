import numpy as np
import pandas as pd
from scipy.stats import norm
import pandas_datareader.data as web
import datetime

#if we want to calculate VaR in a day's time
def value_at_risk(position, c, mu, sigma):
	alpha=norm.ppf(1-c)
	var = position*(mu-sigma*alpha)
	return var
	
#to calculate VaR in n days we have to calculate the mean and standard deviation using
#mu = mu*n and sigma=sigma*sqrt(n) 
def value_at_risk_long(S, c, mu, sigma,n):
	alpha=norm.ppf(1-c)
	var = S*(mu*n-sigma*alpha*np.sqrt(n))
	return var
	

	
#historical data to approximate mean and standard deviation
start_date = datetime.datetime(2014,1,1)
end_date = datetime.datetime(2017,10,15)

#download stock related data from Yahoo Finance
apple = web.DataReader('AAPL',data_source='yahoo',start=start_date,end=end_date)
	
#use pct_change() to calculate the percentage change of daily returns
apple['returns'] = apple['Adj Close'].pct_change()
	
S = 1e6 	#this is the investment in $
c=0.95		#95% confidence level
	
#the underlying assumption is that returns are normally distributed
mu = np.mean(apple['returns'])
sigma = np.std(apple['returns'])

print('Value at risk for tomorrow is: $%0.2f' % value_at_risk(S,c,mu,sigma))
for n in range(10,50,10):
	print(f'Value at risk in {n} days is : $%0.2f' % value_at_risk_long(S,c,mu,sigma,n))

