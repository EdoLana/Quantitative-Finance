import numpy as np
import math
import time
 
class OptionPricing:
    
	def __init__(self,S0,E,T,rf,sigma,iterations):
		#we construct all the attributes needed for the Monte-Carlo simulation
		self.S0 = S0
		self.E = E
		self.T = T
		self.rf = rf
		self.sigma = sigma     
		self.iterations = iterations 
 
	def call_option_simulation(self):
		
		#we have 2 columns: the first one is constantly 0, the second one will store the payoff
		#Note: payoff function is max(0,S-E) for call option
		option_data = np.zeros([self.iterations, 2])
		
		#initialize a normally distributed random array
		rand = np.random.normal(0, 1, [1, self.iterations])
		
		#use the equation for the stock price S(t) 
		stock_price = self.S0*np.exp(self.T*(self.rf - 0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
 
		#calculate S-E
		option_data[:,1] = stock_price - self.E   
        
		#take the average of the the Monte-Carlo method outputs
		
		average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)
 
		#we have to take into account the present value of the money, according to risk-free interest rate
		return np.exp(-1.0*self.rf*self.T)*average
		
	def put_option_simulation(self):
		#we have 2 columns: the first one is constantly 0, the second one will store the payoff
		#Note: payoff function is max(0,E-S) for put option
		option_data = np.zeros([self.iterations, 2])
		
		#initialize a normally distributed random array
		rand = np.random.normal(0, 1, [1, self.iterations])
		
		#use the equation for the stock price S(t) 
		stock_price = self.S0*np.exp(self.T*(self.rf - 0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
 
		#calculate E-S
		option_data[:,1] = self.E - stock_price  
        
		#take the average of the the Monte-Carlo method outputs
		average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)
 
		#we have to take into account the present value of the money, according to risk-free interest rate
		return np.exp(-1.0*self.rf*self.T)*average

	
	#S0 denotes the underlying stock price at t=0, E is the strike price, T os the expiry, rf the risk-free rate
	#sigma is the volatility of the underlying stock
	
model = OptionPricing(S0 = 100,E = 100,T = 1,rf = 0.05,sigma=0.2,iterations = 1000000)
print("Call option price with Monte-Carlo approach: ", model.call_option_simulation()) 
print("Put option price with Monte-Carlo approach: ", model.put_option_simulation())
