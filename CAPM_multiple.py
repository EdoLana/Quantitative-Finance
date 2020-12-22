import pandas_datareader as pdr
import pandas_datareader.data as web 
from datetime import date
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

risk_free_rate = 0.05

def capm(start_date,end_date, stocks, data2):

	#get data
	stock1 = web.DataReader(stocks, data_source = 'yahoo', start = start_date, end = end_date)['Adj Close']
	stock2 = web.DataReader(data2,data_source = 'yahoo', start = start_date, end = end_date)['Adj Close']
	#convert to monthly returns (~normally distributed)
	return_stock1 = stock1.resample('M').last()
	return_stock2 = stock2.resample('M').last()
	data = pd.concat([return_stock1, return_stock2], axis=1, join='inner')

	
	# #normalize data taking logarithms

	data = np.log(data/data.shift(1))

	#remove rows missing data

	data = data.dropna()

	#randomly initialize weights

	weights = np.random.random(len(data.columns)-1)
	weights /= np.sum(weights)

	# #calculate covariance matrix
	covmat = np.cov(data[stocks].T,data['^GSPC'])
	#print(covmat[:-1,-1])

	# #calculate beta

	beta = np.dot(covmat[:-1,-1],weights)/covmat[-1,-1]
	print(f'Beta = {beta}')

	#fit a linear regression to then compare coefficients
	weigthed_stocks = np.dot(weights,data[stocks].T).T
	#print(weigthed_stocks.shape)
	beta,alpha = np.polyfit(data['^GSPC'],weigthed_stocks,deg = 1)
	print(f'Beta from regression: {beta}')

	#plot data

	fig,axis = plt.subplots(1,figsize=(20,10))
	axis.scatter(data['^GSPC'],weigthed_stocks,label = 'Data points')
	axis.plot(data['^GSPC'],beta*data['^GSPC'] + alpha,color = 'red', label = 'CAPM line')
	plt.title('Capital Asset Pricing Model')
	plt.xlabel('Market return $R_m')
	plt.ylabel('Stock return $R_a')
	plt.text(0.08,0.05,r'$R_a = \beta * R_m + \alpha$',fontsize = 18)
	plt.legend( )
	plt.grid(True)
	plt.show()


	# #use CAPM formula to calculate the expected return
	expected_return = risk_free_rate + beta*(data['^GSPC'].mean()*12-risk_free_rate)
	print(f'The expected return is: {expected_return}')

#run the script with data from 2010 to 2016, for a portfolio of 6 stocks, using S&P500
capm('2010-01-01','2017-01-01',['AAPL','WMT','TSLA','GE','AMZN','DB'],['^GSPC'])