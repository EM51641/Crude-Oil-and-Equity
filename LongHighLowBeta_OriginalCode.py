from datetime import datetime

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Indicators")
AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *
from QuantConnect.Orders import *
from QuantConnect.Securities import *
from QuantConnect.Python import PythonData
import decimal
import numpy as np
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import minimize

class CrudeOilPredictsEqeuityReturns(QCAlgorithm):
    
    def Initialize(self):
        # Set the cash we'd like to use for our backtest,we took a big number in order to access to the institutional fees 
        self.SetCash(5000000)

        # Start and end dates for the backtest.
        self.SetStartDate(2000, 1, 1)

        # Add assets we'd like to incorporate into our portfolio
        self.oil = self.AddData(QuandlOil, 'FRED/DCOILWTICO', Resolution.Daily).Symbol #Extract WTI Prices 
        self.TLT = self.AddEquity('TLT', Resolution.Daily).Symbol #(Not included) and can be useful if we want to include a leverage
        self.CPI = self.AddData(QuandlCPI, 'RATEINF/CPI_USA', Resolution.Daily).Symbol#Extract CPI indexes
        self.GDP = self.AddData(QuandlGDP, 'FRED/GDP', Resolution.Daily).Symbol#Extract GDP indexes
        
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.nextLiquidate = self.Time   # Initialize last trade time
        self.nextLiquidate_M = self.Time
        self.rebalance_days = 360 #Yearly Universe Rebalancement 
        self.UniverseSettings.Resolution = Resolution.Daily #Daily computation
        self.AddUniverse(self.CoarseSelection,self.FineSelectionFunction)
        self.selectedequity = 1000 #First Filter
        self.count = 500#Second Filter
        self.symbols = []
        self.Portfolio.MarginModel = PatternDayTradingMarginModel()
        self.rebalance_days_M = 30 #Monthly Portfolio Rebalancing 
        self.lookback = 252 * 5 # 5 years Data
        self.yPred = []
        self.Lens = 0 #Initialise the lenght  of the portfolio
        self.idx = [] #Initialise the Universe List

    def CustomSecurityInitializer(self, security):
        security.SetLeverage(4) #relevant only for the Long Short Portfolio in order to allow sometimes a little more than 1 unit in leverage,otherwise we would had situations when we are stucked.
        security.SetDataNormalizationMode(DataNormalizationMode.Raw) #Use raw data in order to be the most realistic 
        
    def CoarseSelection(self, coarse): #1st Filter
        
        if self.Time < self.nextLiquidate:
            return Universe.Unchanged
        
        selected = sorted([x for x in coarse if x.HasFundamentalData and x.Price > 5],
                          key=lambda x: x.DollarVolume, reverse=True)

        return [x.Symbol for x in selected[:self.selectedequity ] ]
        
    def FineSelectionFunction(self, fine): #2nd Filter
        
        filteredFine = [x for x in fine if x.CompanyReference.IndustryTemplateCode == "T" or x.CompanyReference.IndustryTemplateCode == "N" or x.CompanyReference.IndustryTemplateCode == "M" ]

        self.symbols = [x.Symbol for x in filteredFine][:self.count]
        
        return self.symbols  
    
    def OnData(self,data):
        
        if self.Time >= self.nextLiquidate:
            
            try:
                hist,crudeoil_history = self.Data_Reframe() 
            except:
                self.nextLiquidate = self.Time + timedelta(10) #in case of information breach
            
            self.yPred = self.Beta_Scoring(hist,crudeoil_history).dropna().sort_values(by=['Prediction'])
            
            self.yPred_H = self.yPred[-25:] #Select How many High Beta stocks to choose
            
            self.yPred_L = self.yPred[:25] #Select How many Low Beta stocks to choose
            
            self.yPred = (self.yPred_H).append(self.yPred_L) # Merge Both
            
            self.idx = self.yPred.index.tolist()
            
            for holding in self.Portfolio.Values: # Yearly Rebuilt universe,Liquidate Old Stocks that are not in the new universe
                if holding.Symbol in self.idx  :
                    continue
                if holding.Invested:
                    self.Liquidate(holding.Symbol)
        
            self.Lens = len(self.yPred)
            
            self.nextLiquidate = self.Time + timedelta(self.rebalance_days)
            
            return 
        
        if self.Time < self.nextLiquidate_M:
            
            return

        Monthly_History_X = self.History(self.idx,self.lookback, Resolution.Daily).close.unstack(level=0).resample('1M').last().dropna(axis=1) #60 months stock prices 
        
        Monthly_History_X = Monthly_History_X.pct_change().dropna()
        
        sample_length = len(Monthly_History_X.columns)
        
        init_Weights = np.ones(sample_length)/sample_length
        
        try:
            weights = self._get_risk_parity_weights(init_Weights,Monthly_History_X,Monthly_History_X.mean())
        except:
            m = 1
            self.nextLiquidate = self.Time + timedelta(m) #in case of bug
            return 
        
        weights = pd.DataFrame(weights,index = Monthly_History_X.columns)
        
        #Place Orders when the Market Opens
        for i in weights.index:
            self.SetHoldings(i,weights.loc[i])
        
        #Rebalance the Portfolio Monthly
        self.nextLiquidate_M = self.Time + timedelta(self.rebalance_days_M)
        
    def Beta_Scoring(self,asset_hist,crude_oil_hist) :
        
        factors = sm.add_constant(crude_oil_hist) #add constant
        
        OLSmodels = {ticker: sm.OLS(asset_hist[ticker], factors).fit() for ticker in asset_hist.columns} #Regress on the 3 variables
        
        pvalue = pd.DataFrame({ticker: model.pvalues[3] for ticker, model in OLSmodels.items()},index=["P-values"]).T #Get the p-val associated with the oil sensibility
        
        retained_tickers = pvalue[pvalue < 0.10].dropna().index #Remove unsignificant stocks(at the scale of the Oil sensibility)
        
        Beta = pd.DataFrame({ticker : model.params[3] for ticker , model in OLSmodels.items()},index=["Prediction"]).T #Get the sensibility coefficient (at the scale of the Oil sensibility)
        
        Beta = Beta.loc[retained_tickers] #Keep significant Betas
    
        return Beta
        
    def Data_Reframe(self):
        
        hist = self.History(self.symbols, 1260, Resolution.Daily).close.unstack(level=0).dropna(axis=1).resample('1M').last() #60 months stock prices 
        
        CPI = self.History(QuandlCPI,self.CPI , 2300, Resolution.Daily).droplevel(level=0) # 60 months CPI index
        
        CPI = CPI[~CPI.index.duplicated(keep='last')].resample('1M').last().fillna(method='ffill')
        
        CPI.columns = ["CPI"]
        
        GDP = self.History(QuandlGDP,self.GDP , 2300, Resolution.Daily).droplevel(level=0) # 60 months GDP index
        
        GDP = GDP[~GDP.index.duplicated(keep='last')].resample('1M').last().fillna(method='ffill')
    
        GDP.columns = ["GDP"]
        
        histb = hist.index[0]
        
        histend = hist.index[-1]
        
        L1 = CPI.join(GDP).loc[histb:histend] #Order data as we mix between Quandl and QuantConnect data base
        
        crudeoil_history = self.History(QuandlOil,self.oil,1800, Resolution.Daily).droplevel(level=0) # Import the WTI crude oil price Data
        
        hist = crudeoil_history[~crudeoil_history.index.duplicated(keep='last')].resample('1M').last().join(hist).loc[histb:histend].dropna()
        
        hist = L1.join(hist).loc[histb:histend].dropna() # Remove unsuperposed data
        
        crudeoil_history = hist.iloc[:,:3]
        
        hist = hist.iloc[:,3:]
        
        hist,crudeoil_history = self.Log_returns(hist,crudeoil_history) #Compute log returns of the inputs
        
        crudeoil_history.index = hist.index #Make sure everything goes well,otherwise the algorithm is stopped
        
        return hist,crudeoil_history
        
    def Log_returns(self,hist,crudeoil_history):
        hist = np.log(hist/hist.shift(1)).dropna()
        crudeoil_history = np.log(crudeoil_history/crudeoil_history.shift(1)).dropna()
        return hist,crudeoil_history
    
    def Covariance_Matrice(self,X):
        cov = X.cov()*12
        return cov

    def Margin_contribution(self,Weights,X):
        cov = self.Covariance_Matrice(X)
        Portfolio_risk = np.sqrt(np.dot(np.dot(Weights.T,cov),Weights))
        Margin_cs = np.dot(cov,Weights)/Portfolio_risk
        return Margin_cs,Portfolio_risk

    def Total_Risk_Contribution_errors(self,Weights,args):
        X = args[0]
        budgetting_weights = args[1]
        Margin_cs,portfolio_risk =  self.Margin_contribution(Weights,X)
        RC_taken = np.square(Weights*Margin_cs/self.Lens - budgetting_weights*portfolio_risk).sum()#portfolio_risk
        return RC_taken
        
    def Min_Var(self,w,args):
        X = args[0]
        cov = self.Covariance_Matrice(X)
        return (np.dot(w.T,np.dot(cov,w))) #Covariance Matrice
        
    def Mean_Variance(self,w,args): #Optional(not included in the paper)
        X = args[0]
        mu = args[2]
        cov = self.Covariance_Matrice(X)
        p_variance = np.dot(w,np.dot(cov,w.T))
        returns = np.dot(w,mu.T)
        LK = returns-(5/2)*p_variance
        return -LK
        
    def most_div(self,w,args): #Optional(not included in the paper)
        X = args[0]
        cov = self.Covariance_Matrice(X)
        std = np.sqrt(np.diag(cov))
        w_volatility = np.dot(w,std)
        p_volatility = np.sqrt(np.dot(w,np.dot(cov,w.T)))
        diversification_ratio = w_volatility/p_volatility
        return -diversification_ratio
        
    def expected_return(self,w,args): #Optional(not included in the paper)
        retrn = args[0] * 12
        return np.dot(w,retrn)-0.10
    
    def _get_risk_parity_weights(self,init_Weights,X,Monthly_History_X): #Optimisation fuction
    
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1}) #Add -1 for Long only portfolios
                       
                       
        bnds = tuple((0.01,np.maximum(0.10,init_Weights[0]+0.01)) for s in init_Weights) #Replace left array by 1% for Long only Portfolio
    
        optimize_result = minimize(fun=self.Min_Var, #Replace Min_Var by Total_Risk_Contribution_errors if you want to switch from Min Variance to Risk Parity
                               args=[X,init_Weights,Monthly_History_X],
                               x0=init_Weights,
                               bounds = bnds,
                               method='SLSQP',
                               tol=1e-04,
                               constraints=constraints,
                               options={'disp': False})

        weights = optimize_result.x
    
        weights = weights 
        
        return weights  # It returns the optimised weights

    def Leverage_to_use(self,weights,X,Backward_Period ): #Optional (not included in the paper) , add Leverage Function to the optimiser if you want to increase the portfolio leverage
        Monthly_History_VTI = self.History(self.TLT, self.lookback, Resolution.Daily).close.unstack(level=0).resample('1M').last().pct_change().dropna()
        used_metric = X[-Backward_Period:]
        market_std = Monthly_History_VTI.iloc[-Backward_Period:].std() * 12 **(1/2) 
        Portfolio_pct_change = (weights*used_metric).sum(axis=1)
        Portfolio_std = Portfolio_pct_change.std() * 12**(1/2)
        k = np.minimum((market_std/Portfolio_std).values,2)
        return k #Leverage Multiplier
            
class QuandlOil(PythonQuandl):
    def __init__(self):
        self.ValueColumnName = 'Value'
        
class QuandlCPI(PythonQuandl):
    def __init__(self):
        self.ValueColumnName = 'Value'
        
class QuandlGDP(PythonQuandl):
    def __init__(self):
        self.ValueColumnName = 'Value'
