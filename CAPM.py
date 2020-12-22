import pandas_datareader as pdr
from pandas_datareader import data,wb
from datetime import date
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

risk_free_rate = 0.05

def capm(start_date,end_date, data1, data2):

	#get data
	stock1 = pdr.get_data_yahoo(data1,start_date,end_date)
	stock2 = pdr.get_data_yahoo(data2,start_date,end_date)
	#convert to monthly returns (~normally distributed)
	return_stock1 = stock1.resample('M').last()
	return_stock2 = stock2.resample('M').last()

	#create a DataFrame using the adjusted close pricing
	data = pd.DataFrame({'s_adjclose':return_stock1['Adj Close'], 'm_adjclose' : return_stock2['Adj Close']},index = return_stock1.index)

	#normalize data taking logarithms
	data[['s_returns','m_returns']] = np.log(data[['s_adjclose','m_adjclose']]/data[['s_adjclose','m_adjclose']].shift(1))

	#remove rows missing data

	data = data.dropna()

	#calculate covariance matrix

	covmat = np.cov(data['s_returns'],data['m_returns'])

	#calculate beta

	beta = covmat[0,1]/covmat[1,1]
	print(f'Beta = {beta}')

	#fit a linear regression to then compare coefficients

	beta,alpha = np.polyfit(data['m_returns'],data['s_returns'],deg = 1)
	print(f'Beta from regression: {beta}')

	#plot data

	fig,axis = plt.subplots(1,figsize=(20,10))
	axis.scatter(data['m_returns'],data['s_returns'],label = 'Data points')
	axis.plot(data['m_returns'],beta*data['m_returns'] + alpha,color = 'red', label = 'CAPM line')
	plt.title('Capital Asset Pricing Model')
	plt.xlabel('Market return $R_m')
	plt.ylabel('Stock return $R_a')
	plt.text(0.08,0.05,r'$R_a = \beta * R_m + \alpha$',fontsize = 18)
	plt.legend( )
	plt.grid(True)
	plt.show()


	#use CAPM formula to calculate the expected return
	expected_return = risk_free_rate + beta*(data['m_returns'].mean()*12-risk_free_rate)
	print(f'The expected return is: {expected_return}')

#run the script with data from 2010 to 2016, for the IBM stock, using S&P500
capm('2010-01-01','2017-01-01','IBM','^GSPC')