##Code made to run on QuantConnect.com

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
        self.SetCash(100000)

        # Start and end dates for the backtest.
        self.SetStartDate(2003, 1, 1)

        # Add assets we'd like to incorporate into our portfolio
        self.oil = self.AddData(QuandlOil, 'FRED/DCOILBRENTEU', Resolution.Daily).Symbol
        self.IEF = self.AddEquity('IEF', Resolution.Daily).Symbol
        self.AddData(TBill, "tbill")
        self.tbill = self.Securities["tbill"].Symbol
        self.nextLiquidate = self.Time   # Initialize last trade time
        self.rebalance_days = 30
        self.UniverseSettings.Resolution = Resolution.Daily
        self.selectedequity = 200
        self.symbols = []
        self.Portfolio.MarginModel = PatternDayTradingMarginModel()
        self.Count_liquidations = 0
        self.AddUniverse(self.Universe.Index.QC500)
        
    def OnData(self,data):
        
        if self.Time < self.nextLiquidate:
            return 
        
        hist,crudeoil_history = self.Data_Reframe()
        
        rf = float(self.Securities[self.tbill].Price)/12.0
        
        yPred = self.Beta_Scoring(hist,crudeoil_history)
        
        yPred_Long = yPred.dropna().sort_values(by=['Prediction'])[:25]
        
        yPred_Short = yPred.dropna().sort_values(by=['Prediction'])[-25:]
        
        Long,Short = len(yPred_Long.index),len(yPred_Short.index)

        for holding in self.Portfolio.Values:
            if holding.Symbol in yPred_Long.index or holding.Symbol in yPred_Short.index or holding.Symbol == self.IEF:
                continue
            if holding.Invested:
                self.Count_liquidations = + 1
                self.Liquidate(holding.Symbol)
        self.Debug(self.Count_liquidations)
    
        for i in yPred_Long.index:
            self.SetHoldings(i, -0.50/Long) 
            
        for i in yPred_Short.index:
            self.SetHoldings(i, 0.50/Short) 
            
        self.SetHoldings(self.IEF, 0.95) 
            
        self.nextLiquidate = self.Time + timedelta(self.rebalance_days)
        
        
    def Beta_Scoring(self,asset_hist,crude_oil_hist) :
        
        factors = sm.add_constant(crude_oil_hist)
        
        OLSmodels = {ticker: sm.OLS(asset_hist[ticker], factors).fit() for ticker in asset_hist.columns}
        
        pvalue = pd.DataFrame({ticker: model.pvalues[1] for ticker, model in OLSmodels.items()},index=["P-values"]).T
        
        retained_tickers = pvalue[pvalue < 0.05].dropna().index
        
        Beta = pd.DataFrame({ticker : model.params[1] for ticker , model in OLSmodels.items()},index=["Prediction"]).T
        
        Beta = Beta.loc[retained_tickers]
        
        New_filter = Beta.dropna().index
    
        Beta = Beta.loc[New_filter]
    
       # Unsystematic_Var = pd.DataFrame({ticker: (model.resid).var()*12  for ticker , model in OLSmodels.items()},index=["Prediction"]).T
        
        #self.Debug(Unsystematic_Var)
        
        #Unsystematic_Var = Unsystematic_Var.loc[New_filter]
        
        Garch_forecast_variance = self.Garch_volatility(crude_oil_hist)
        
        Predictions = (Beta**2)*Garch_forecast_variance  #Unsystematic_Var
    
        return Predictions
        
    def Data_Reframe(self):
        
        hist = self.History(self.symbols, 1800, Resolution.Daily).close.unstack(level=0).dropna(axis=1).resample('1M').last()
        
        rd = len(hist)
        
        crudeoil_history = self.History(QuandlOil,self.oil , 3000, Resolution.Daily).droplevel(level=0)
        
        crudeoil_history = crudeoil_history[~crudeoil_history.index.duplicated(keep='last')].resample('1M').last().iloc[-rd:]
        
        hist,crudeoil_history = self.Log_returns(hist,crudeoil_history)
        
        crudeoil_history.index = hist.index
        
        return hist,crudeoil_history
        
    def Log_returns(self,hist,crudeoil_history):
        hist = np.log(hist/hist.shift(1)).dropna()
        crudeoil_history = np.log(crudeoil_history/crudeoil_history.shift(1)).dropna()
        return hist,crudeoil_history
        
    def Garch_volatility(self,crudeoil_history):
        am = arch_model(crudeoil_history*100, p=1, o=1, q=1,mean='AR')
        res = am.fit(update_freq=1)
        forecast_var = res.forecast(horizon=1)
        forecast_var = (forecast_var.variance.iloc[-1][0]/(100**2))*12**(1/2)
        return forecast_var
        
    def OnSecuritiesChanged(self, changes):
        
        self.symbols = [x.Symbol for x in changes.AddedSecurities]
        
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol, 'Removed from Universe')

class TBill(PythonData):
    
    def GetSource(self, config, date, isLiveMode):
        # Get the data source from Quandl
        # Ascending order of the data file is essential!
        return SubscriptionDataSource("https://www.quandl.com/api/v3/datasets/USTREASURY/BILLRATES.csv?api_key=zxb6rfszSQW5-SLkaj3t&order=asc", SubscriptionTransportMedium.RemoteFile)
    
    def Reader(self, config, line, date, isLiveMode):
        tbill = TBill()
        tbill.Symbol = config.Symbol
        
        # Example Line Format:
        # Date      4 Wk Bank Discount Rate   
        # 2017-06-01         0.8    
        if not (line.strip() and line[0].isdigit()): return None
        
        # Parse the file based on how it is organized
        try:
            data = line.split(',')
            value = float(data[1])*0.01
            value = decimal.Decimal(value)
            if value == 0: return None
            tbill.Time = datetime.strptime(data[0], "%Y-%m-%d")
            tbill.Value = value
            tbill["Close"] = float(value)
            return tbill;
        except ValueError:
            return None
            
class QuandlOil(PythonQuandl):
    def __init__(self):
        self.ValueColumnName = 'Value'
