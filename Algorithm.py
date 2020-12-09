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

class CrudeOilPredictsEqeuityReturns(QCAlgorithm):
    
    def Initialize(self):
        # Set the cash we'd like to use for our backtest
        self.SetCash(1000000) #We have chosen this number in order to have access to premium slippage fees

        # Start and end dates for the backtest.
        self.SetStartDate(2003, 1, 1)

        # Add assets we'd like to incorporate into our portfolio
        self.oil = self.AddData(QuandlOil, 'FRED/DCOILBRENTEU', Resolution.Daily).Symbol
        self.IEF = self.AddEquity('IEF', Resolution.Daily).Symbol
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.nextLiquidate = self.Time   # Initialize last trade time
        self.rebalance_days = 30
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelection,self.FineSelectionFunction)
        self.selectedequity = 1000
        self.count = 150
        self.symbols = []
        self.Portfolio.MarginModel = PatternDayTradingMarginModel()
        self.Count_liquidations = 0
        
    def CustomSecurityInitializer(self, security):
        security.SetLeverage(4)
        security.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
    def CoarseSelection(self, coarse):
        
        if self.Time < self.nextLiquidate:
            return Universe.Unchanged
        
        selected = sorted([x for x in coarse if x.HasFundamentalData and x.Price > 5],
                          key=lambda x: x.DollarVolume, reverse=True)

        return [x.Symbol for x in selected[:self.selectedequity ] ]
        
    def FineSelectionFunction(self, fine):
        
        filteredFine = [x for x in fine if x.CompanyReference.IndustryTemplateCode == "T" ]

        self.symbols = [x.Symbol for x in filteredFine][:self.count]
        
        return self.symbols 
    
    def OnData(self,data):
        
        if self.Time < self.nextLiquidate:
            return 
        
        hist,crudeoil_history = self.Data_Reframe()
        
        yPred = self.Beta_Scoring(hist,crudeoil_history)
        
        yPred_Long = yPred.dropna().sort_values(by=['Prediction'])[:20]
        
        yPred_Short = yPred.dropna().sort_values(by=['Prediction'])[-20:]
        
        Long,Short = len(yPred_Long.index),len(yPred_Short.index)

        for holding in self.Portfolio.Values:
            if holding.Symbol in yPred_Long.index or holding.Symbol in yPred_Short.index or holding.Symbol == self.IEF:
                continue
            if holding.Invested:
                self.Liquidate(holding.Symbol)
    
        for i in yPred_Long.index:
            self.SetHoldings(i, 0.50/Long) 
            
        for i in yPred_Short.index:
            self.SetHoldings(i, -0.50/Short) 
            
        self.SetHoldings(self.IEF, 1) #0.5-0.5+1 = 1
            
        self.nextLiquidate = self.Time + timedelta(self.rebalance_days)
        
        
    def Beta_Scoring(self,asset_hist,crude_oil_hist) :
        
        factors = sm.add_constant(crude_oil_hist)
        
        OLSmodels = {ticker: sm.OLS(asset_hist[ticker], factors).fit() for ticker in asset_hist.columns}
        
        pvalue = pd.DataFrame({ticker: model.pvalues[1] for ticker, model in OLSmodels.items()},index=["P-values"]).T
        
        retained_tickers = pvalue[pvalue < 0.10].dropna().index #Only significant parameters interest us
        
        Beta = pd.DataFrame({ticker : model.params[1] for ticker , model in OLSmodels.items()},index=["Prediction"]).T
        
        Beta = Beta.loc[retained_tickers]
        
        Garch_forecast_variance = self.Garch_volatility(crude_oil_hist)
        
        Predictions = (Beta**2)*Garch_forecast_variance #We should add up a 1 as in an AR(1) residuals are independentely distributed.However as it won't change anything in the ranking we didn't add it.
    
        return Predictions
        
    def Data_Reframe(self):
        
        hist = self.History(self.symbols, 252, Resolution.Daily).close.unstack(level=0).dropna(axis=1)
        
        histb = hist.index[0]
        
        histend = hist.index[-1]
        
        crudeoil_history = self.History(QuandlOil,self.oil , 500, Resolution.Daily).droplevel(level=0)
        
        hist = crudeoil_history[~crudeoil_history.index.duplicated(keep='last')].join(hist).loc[histb:histend].dropna()
        
        crudeoil_history = hist.iloc[:,:1]
        
        hist = hist.iloc[:,1:]
        
        hist,crudeoil_history = self.Log_returns(hist,crudeoil_history)
        
        crudeoil_history.index = hist.index #Make sure everything is fine
        
        return hist,crudeoil_history
        
    def Log_returns(self,hist,crudeoil_history):
        hist = np.log(hist/hist.shift(1)).dropna()
        crudeoil_history = np.log(crudeoil_history/crudeoil_history.shift(1)).dropna()
        return hist,crudeoil_history
        
    def Garch_volatility(self,crudeoil_history):
        am = arch_model(crudeoil_history*100, p=1, o=1, q=1,mean='AR')
        res = am.fit(update_freq=1)
        forecast_var = res.conditional_volatility.iloc[-1]
        return forecast_var
            
class QuandlOil(PythonQuandl):
    def __init__(self):
        self.ValueColumnName = 'Value'
