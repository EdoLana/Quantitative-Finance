import pandas_datareader as pdr
from pandas_datareader import data, wb
from datetime import date
import numpy as np
import pandas as pd
from scipy import log,exp,sqrt,stats

def blackscholes_call(S,E,T,rf,sigma):
	#calculate d1 and d2 parameters
	d1=(log(S/E)+(rf+sigma*sigma/2.0)*T)/(sigma*sqrt(T))
	d2 = d1-sigma*sqrt(T)
	print(d1)
	print(d2)
	#we use the cdf of a standard normal distribution 
	return S*stats.norm.cdf(d1)-E*exp(-rf*T)*stats.norm.cdf(d2)

def blackscholes_put(S,E,T,rf,sigma):
	#calculate d1 and d2 parameters
	d1=(log(S/E)+(rf+sigma*sigma/2.0)*T)/(sigma*sqrt(T))
	d2 = d1-sigma*sqrt(T)
	#we use the cdf of a normal distribution
	return -S*stats.norm.cdf(-d1)+E*exp(-rf*T)*stats.norm.cdf(-d2)
	

#In what follows, S0 denotes the underlying stock price at time t = 0, E is the strike/exercise price, T is the expiry date
#(1 year), rf is the risk-free rate and sigma is the volatility of the underlying stock.

print("Call option price according to Black-Scholes model: ",blackscholes_call(S0 = 100,E = 100,T=1,rf=0.05,sigma=0.2))
print("Put option price according to Black-Scholes model: ",blackscholes_put(S0 = 100,E = 100,T=1,rf=0.05,sigma=0.2))