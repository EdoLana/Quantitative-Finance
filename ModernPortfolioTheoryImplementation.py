import numpy as np 
import pandas_datareader.data as web 
import matplotlib.pyplot as plt 
import scipy.optimize as optimization

#choose some stocks
stocks = ['AAPL','WMT','TSLA','GE','AMZN','DB']


start_date = '01/01/2001'
end_date = '01/01/2019'

#Download data from Yahoo Finance
def download_data(stocks):
	data = web.DataReader(stocks, data_source = 'yahoo', start = start_date, end = end_date)['Adj Close']
	return data

# daily_returns = (data/data.shift(1)) - 1   #formula for daily return
# daily_returns.hist(bins = 100)
# plt.show() 								#check that daily returns are normally distributed


def show_data(data):
	data.plot(figsize = (10,5))
	plt.show()

#normalize using natural logarithm

def calculate_returns(data):
	returns = np.log((data/data.shift(1)))     		#we do not subtract one to keep the argument close to 1
	return returns

def plot_daily_returns(returns):
	returns.plot(figsize=(10,5))
	plt.show()
#print mean and covariance matrix of the stocks within [start_date,end_date] interval time.
def show_statistics(returns):						# Note that there are 252 trading days in a year	
	print(returns.mean())
	print(returns.cov())

#define random weights
def initialize_weights():
	weights = np.random.random(len(stocks))
	weights /= np.sum(weights)
	return weights
data = download_data(stocks)
returns = calculate_returns(data)
weights = initialize_weights()

# plot_daily_returns(returns)
# show_statistics(returns)						
# print(returns)
# print(weights)

#expected portfolio returns

def calculate_portfolio_return(returns,weights):
	portfolio_return = np.sum(returns.mean()*weights)*252
	#print(f'The expected portfolio return is: {portfolio_return}')
	return portfolio_return

#expected portfolio variance

def calculate_portfolio_stdev(returns,weights):
	portfolio_stdev = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
	#print(f'The expected portfolio standard deviation is: {portfolio_stdev}')
	return portfolio_stdev

# calculate_portfolio_return(returns,weights)
# calculate_portfolio_stdev(returns,weights)

#generate random portfolios by using Monte-Carlo simulation

def generate_portfolios(returns):

	preturns = []
	pstdev = []

	for i in range(2*(10**4)):
		weights = initialize_weights()
		preturns.append(calculate_portfolio_return(returns,weights))
		pstdev.append(calculate_portfolio_stdev(returns,weights))

	preturns = np.array(preturns)
	pstdev = np.array(pstdev)
	return preturns,pstdev

#scatterplot of sharpe ratio
preturns,pstdevs = generate_portfolios(returns)
# def plot_portfolios_Sharperatio(returns,stdevs):
# 	plt.figure(figsize = (10,6))
# 	plt.scatter(stdevs,returns, c = returns/stdevs, marker = 'o')
# 	plt.grid(True)
# 	plt.xlabel('Expected volatility')
# 	plt.ylabel('Expected return')
# 	plt.colorbar(label = 'Sharpe ratio')
# 	plt.show()

# plot_portfolios_Sharperatio(preturns,pstdevs)

#let's group together the statistics we are interested in
def statistics(weights,returns):
	portfolio_return = calculate_portfolio_return(returns,weights)
	portfolio_stdev = calculate_portfolio_stdev(returns,weights)
	return np.array([portfolio_return,portfolio_stdev,portfolio_return/portfolio_stdev])

#define the object function to minimize
def min_func(weights,returns):
	return -statistics(weights,returns)[2]

#use numerical optimization to obtain the best portfolio

def optimize_portfolio(weights,returns):
	constraints = ({'type': 'eq','fun': lambda x : np.sum(x) -1 })  #our constraint is that the weights add up to 1
	bounds = tuple((0,1) for x in range(len(stocks)))  #set the range in which the weights live
	optimum = optimization.minimize(fun = min_func, x0 = weights, args = returns,method = 'SLSQP',bounds = bounds, constraints = constraints)
	return optimum

#display the optimal portfolio
def print_optimal_portfolio(optimum,returns):
	print(f'Optimal weights: {optimum.x.round(3)}')
	print(f'Expected return, volatility and Sharpe ratio: {statistics(optimum.x.round(3),returns)}')

#locate the optimal portfolio on the scatterplot
def show_optimal_portfolio(optimum,returns,preturns,pstdevs):
	plt.figure(figsize = (10,6))
	plt.scatter(pstdevs,preturns, c = preturns/pstdevs, marker = 'o')
	plt.grid(True)
	plt.xlabel('Expected volatility')
	plt.ylabel('Expected return')
	plt.colorbar(label = 'Sharpe ratio')
	plt.plot(statistics(optimum.x,returns)[1],statistics(optimum.x,returns)[0],'g*', markersize = 20.0)
	plt.show()

optimum = optimize_portfolio(weights,returns)
print_optimal_portfolio(optimum,returns)
show_optimal_portfolio(optimum,returns,preturns,pstdevs)