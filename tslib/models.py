from statsmodels.tsa.arima.model import ARIMA
import warnings
import statsmodels.api as sm    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from prophet import Prophet
from tbats import TBATS
from statsforecast.models import ADIDA, CrostonOptimized,TSB,IMAPA
from tslib.interface import TSModel
from tslib.exceptions import (NoDiagnosticsModelAbsent,
                            NoForecastModelAbsent,
                            NoSummaryModelAbsent,
                            NoPlotModelAbsent,NoConfidenceIntervalModelAbsent)
warnings.simplefilter('ignore', category=UserWarning)

class Arima(TSModel):
    def __init__(self):
        self.m = None
        self.fit_options = None
        self.ts=None
        self.fitted_values=None
        self.model_init_opts = None

    def fit(self,ts,**kwargs):
        m = ARIMA(ts,**kwargs)
        m = m.fit()
        self.m = m
        self.fit_options = kwargs
        self.ts=ts
        self.fitted_values=self.m.fittedvalues
        return self 
    
    def summary(self):
        if self.m is not None:
            return self.m.summary()
        else:
            raise NoSummaryModelAbsent

    def forecast(self,steps=5,**kwargs):
        if self.m is not None:
            return self.m.forecast(steps=steps,**kwargs)
        else:
            raise NoForecastModelAbsent

    def residual_analysis(self):
        if self.m is not None:
            return self.m.plot_diagnostics()
        else:
            raise NoDiagnosticsModelAbsent
    def plot_fit(self):
        if self.m is not None and self.ts is not None:
            x_range = self.ts.index
            fitted_vals = self.m.fittedvalues 
            actual_vals = self.ts
            fig,ax = plt.subplots(figsize=(5,5))
            ax.plot(x_range,actual_vals,label='Actual values',color='black')
            ax.plot(x_range,fitted_vals,label='Fitted values',color='red')
            ax.set_title('Actual vs Fitted Plot')
            plt.plot() 

        else:
            raise NoPlotModelAbsent
    def get_fit_interval(self,alpha=0.05):
        if self.m is not None:
            p = self.m.get_prediction()
            return p.conf_int(alpha=alpha)
        else:
            raise NoConfidenceIntervalModelAbsent
    def get_forecast_interval(self,steps=5,alpha=0.05):
        if self.m is not None:
            f = self.m.get_forecast(steps=steps)
            return f.conf_int(alpha=alpha)
        else:
            raise NoConfidenceIntervalModelAbsent

class AutoArima(TSModel):
    def __init__(self):
        self.m=None
        self.fit_options=None
        self.fitted_values=None
        self.model_init_opts = None
    
    def fit(self,ts,**kwargs):
        m = pm.auto_arima(ts,**kwargs)
        self.m = m
        self.fit_options = kwargs
        self.ts = ts
        self.fitted_values=self.m.fittedvalues
        return self
    
    def summary(self):
        if self.m is not None:
            return self.m.summary()
        else:
            raise NoSummaryModelAbsent
    
    def forecast(self,steps=5,**kwargs):
        if self.m is not None:
            return self.m.predict(n_periods=steps,**kwargs)
        else:
            raise NoForecastModelAbsent
    
    def residual_analysis(self):
        if self.m is not None:
            self.m.plot_diagnostics()
        else:
            raise NoDiagnosticsModelAbsent
    def plot_fit(self):
        if self.m is not None and self.ts is not None:
            x_range = self.ts.index
            fitted_vals = self.m.fittedvalues() 
            actual_vals = self.ts
            fig,ax = plt.subplots(figsize=(5,5))
            ax.plot(x_range,actual_vals,label='Actual values',color='black')
            ax.plot(x_range,fitted_vals,label='Fitted values',color='red')
            ax.set_title('Actual vs Fitted Plot')
            plt.plot() 

        else:
            raise NoPlotModelAbsent
    def get_fit_interval(self,alpha=0.05):
        if self.m is not None:
            result = pd.DataFrame()
            _,conf_int = self.m.predict_in_sample(return_conf_int=True)
            result['lower_bound'] = conf_int[:,0]
            result['upper_bound'] = conf_int[:,1]
            return result
        else:
            raise NoConfidenceIntervalModelAbsent
    def get_forecast_interval(self,steps=5,alpha=0.05):
        if self.m is not None:
            result = pd.DataFrame()
            _,conf_int = self.m.predict(n_periods=steps,return_conf_int=True,alpha=alpha)
            result['lower_bound'] = conf_int[:,0]
            result['upper_bound'] = conf_int[:,1]
            return result
        else:
            raise NoConfidenceIntervalModelAbsent

class ProphetModel(TSModel):
    def __init__(self,**kwargs):
        self.m = Prophet(**kwargs) 
        self.fit_options = None
        self.ts= None
        self.fitted_values=None
        self.model_init_opts = kwargs
        
    def fit(self,ts:pd.DataFrame,**kwargs):
        self.m.fit(ts,**kwargs)
        self.ts=ts
        self.fitted_values=self.m.predict()['yhat']
        self.fit_options=kwargs
        return self
    def forecast(self,steps=5)->pd.DataFrame:
        if self.m is None:
            raise NoForecastModelAbsent
        future = self.m.make_future_dataframe(periods=steps)
        frsct = self.m.predict(future)
        return frsct

    def summary(self):
        if self.m is not None and self.ts is not None:
           insample_preds = self.m.predict()
           insample_preds['actual']=self.ts.y
           mae = (insample_preds.actual-insample_preds.yhat).abs().mean()
           cov = (insample_preds.actual>=insample_preds.yhat_lower)&(insample_preds.actual<=insample_preds.yhat_upper)
           cov = cov.mean()*100
           sum = {'mae':mae,'coverage':cov}
           return sum
        else:
           raise NoSummaryModelAbsent
    
    def residual_analysis(self):
        if self.m is not None and self.ts is not None:
            r = self.ts.y-self.m.predict()['yhat']
            fig1 = sm.qqplot(r,line='45')
            fig1.suptitle('QQ plot')
            self.m.plot_components(self.m.predict())
        else:
            raise NoDiagnosticsModelAbsent
    
    def plot_fit(self):
        if self.m is not None:
            frsct = self.m.predict()
            self.m.plot(frsct)
        else:
            raise NoPlotModelAbsent
    
    def get_fit_interval(self):
        if self.m is not None:
            p = self.m.predict()[['yhat_lower','yhat_upper']]
            return p
        else:
            raise NoConfidenceIntervalModelAbsent
    
    def get_forecast_interval(self,steps=5):
        if self.m is not None:
            future = self.m.make_future_dataframe(periods=steps)
            return self.m.predict(future)[['yhat_lower','yhat_upper']]
        else:
            raise NoConfidenceIntervalModelAbsent

class TBATSModel(TSModel):
    def __init__(self,**kwargs):
        self.m = TBATS(**kwargs)
        self.fit_options=None
        self.fitted_values=None
        self.model_init_opts = kwargs
    
    def fit(self,ts:pd.Series,**kwargs):
        self.m = self.m.fit(ts,**kwargs)
        self.ts=ts
        self.fitted_values=self.m.y_hat
        self.fit_options=kwargs
        return self    

    def forecast(self,steps=5)->pd.Series:
        if self.m is None:
            raise NoForecastModelAbsent
        frsct = self.m.forecast(steps=steps)
        return frsct
    
    def summary(self):
        if self.m is None:
            raise NoSummaryModelAbsent
        return self.m.summary()
        

    def residual_analysis(self):
        if self.m is not None and self.ts is not None:
            r=self.m.resid
            fig1 = sm.qqplot(r,line='45')
            fig1.suptitle('QQ plot')
        else:
            raise NoDiagnosticsModelAbsent
    
    def plot_fit(self):
        if self.m is not None and self.ts is not None:
            x_range = self.ts.index
            fitted_vals = self.m.y_hat
            actual_vals = self.ts
            _,ax = plt.subplots(figsize=(5,5))
            ax.plot(x_range,actual_vals,label='Actual values',color='black')
            ax.plot(x_range,fitted_vals,label='Fitted values',color='red')
            ax.set_title('Actual vs Fitted Plot')
            plt.plot() 
    
    def get_fit_interval(self,alpha=0.05):
        if self.m is not None:
            f = self.m._calculate_confidence_intervals(predictions=self.fitted_values,level=alpha)
            result = pd.DataFrame()
            result['lower_bound']=f['lower_bound']
            result['upper_bound']=f['upper_bound']
            return result
        else:
            raise NoConfidenceIntervalModelAbsent
    
    def get_forecast_interval(self,steps=5,alpha=0.05):
        if self.m is not None:
            f = self.m.forecast(steps=steps,confidence_level=alpha)
            result = pd.DataFrame()
            result['lower_bound']=f[1]['lower_bound']
            result['upper_bound']=f[1]['upper_bound']
            return result
        else:
            raise NoConfidenceIntervalModelAbsent
        
class CrostonModel(TSModel):
    def __init__(self,**kwargs):
        self.ts=None
        self.fit_options=None
        self.model_init_opts = kwargs
        self.m=CrostonOptimized(**self.model_init_opts)
        self.fitted_values=None
    
    def fit(self,ts:pd.Series,**kwargs):
        self.ts=ts
        self.fit_options=kwargs
        self.m.fit(ts.values,**kwargs)
        self.fitted_values = self.m.predict_in_sample()['fitted']
        return self 
    
    def forecast(self,steps=5):
        if self.m is not None:
            frsct=self.m.predict(h=steps)['mean']
        else:
            raise NoForecastModelAbsent
        return frsct

    def summary(self):
        raise NotImplementedError 

    def residual_analysis(self):
        if self.m is not None and self.ts is not None:
            r = self.ts-self.fitted_values
            fig1 = sm.qqplot(r,line='45')
            fig1.suptitle('QQ plot')
        else:
            raise NoDiagnosticsModelAbsent       

    def plot_fit(self):
        if self.m is not None and self.ts is not None:
            x_range = self.ts.index
            fitted_vals = self.fitted_values
            actual_vals = self.ts
            _,ax = plt.subplots(figsize=(5,5))
            ax.plot(x_range,actual_vals,label='Actual values',color='black')
            ax.plot(x_range,fitted_vals,label='Fitted values',color='red')
            ax.set_title('Actual vs Fitted Plot')
            plt.plot() 

    def get_fit_interval(self,alpha=0.05):
        if self.m is not None:
            r = self.m.predict_in_sample(level=[100-alpha*100])
            results = pd.DataFrame()
            results['lower_bound']=r[f'fitted-lo-{100-alpha*100}']
            results['upper_bound']=r[f'fitted-hi-{100-alpha*100}']
            return results
        else:
            raise NoConfidenceIntervalModelAbsent
    
    def get_forecast_interval(self,steps=5,alpha=0.05):
        if self.m is not None:
            r = self.m.predict(h=steps,level=[100-alpha*100])
            results = pd.DataFrame()
            results['lower_bound']=r[f'lo-{100-alpha*100}']
            results['upper_bound']=r[f'hi-{100-alpha*100}']
            return results
        else:
            raise NoConfidenceIntervalModelAbsent
        
class ADIDAModel(TSModel):
    def __init__(self,**kwargs):
        self.ts=None 
        self.model_init_opts=kwargs
        self.fit_options=None
        self.fitted_values=None 
        self.m=ADIDA(**kwargs)
    
    def fit(self,ts:pd.Series,**kwargs):
        self.ts=ts
        self.fit_options=kwargs 
        self.m.fit(ts.values,**kwargs)
        self.fitted_values = self.m.predict_in_sample()['fitted']
        return self 
    
    def forecast(self,steps=5):
        if self.m is not None:
            frsct=self.m.predict(h=steps)['mean']
        else:
            raise NoForecastModelAbsent
        return frsct

    def summary(self):
        raise NotImplementedError 
    
    def residual_analysis(self):
        if self.m is not None and self.ts is not None:
            r = self.ts-self.fitted_values
            fig1 = sm.qqplot(r,line='45')
            fig1.suptitle('QQ plot')
        else:
            raise NoDiagnosticsModelAbsent       

    def plot_fit(self):
        if self.m is not None and self.ts is not None:
            x_range = self.ts.index
            fitted_vals = self.fitted_values
            actual_vals = self.ts
            _,ax = plt.subplots(figsize=(5,5))
            ax.plot(x_range,actual_vals,label='Actual values',color='black')
            ax.plot(x_range,fitted_vals,label='Fitted values',color='red')
            ax.set_title('Actual vs Fitted Plot')
            plt.plot()
    
    def get_fit_interval(self,alpha=0.05):
        if self.m is not None:
            r = self.m.predict_in_sample(level=[100-alpha*100])
            results = pd.DataFrame()
            results['lower_bound']=r[f'fitted-lo-{100-alpha*100}']
            results['upper_bound']=r[f'fitted-hi-{100-alpha*100}']
            return results
        else:
            raise NoConfidenceIntervalModelAbsent
    
    def get_forecast_interval(self,steps=5,alpha=0.05):
        if self.m is not None:
            r = self.m.predict(h=steps,level=[100-alpha*100])
            results = pd.DataFrame()
            results['lower_bound']=r[f'lo-{100-alpha*100}']
            results['upper_bound']=r[f'hi-{100-alpha*100}']
            return results
        else:
            raise NoConfidenceIntervalModelAbsent
        
class TSBModel(TSModel):
    def __init__(self,**kwargs):
        self.fitted_values=None
        self.fit_options=None
        self.ts=None
        self.model_init_opts=kwargs
        self.m=TSB(**kwargs)
    
    def fit(self,ts,**kwargs):
        self.ts=ts
        self.fit_options=kwargs
        self.m.fit(ts.values,**kwargs)
        self.fitted_values=self.m.predict_in_sample()['fitted']
        return self

    def forecast(self,steps=5):
        if self.m is not None:
            frsct=self.m.predict(h=steps)['mean']
        else:
            raise NoForecastModelAbsent
        return frsct
    
    def summary(self):
        raise NotImplementedError
    
    def residual_analysis(self):
        if self.m is not None and self.ts is not None:
            r = self.ts-self.fitted_values
            fig1 = sm.qqplot(r,line='45')
            fig1.suptitle('QQ plot')
        else:
            raise NoDiagnosticsModelAbsent       

    def plot_fit(self):
        if self.m is not None and self.ts is not None:
            x_range = self.ts.index
            fitted_vals = self.fitted_values
            actual_vals = self.ts
            _,ax = plt.subplots(figsize=(5,5))
            ax.plot(x_range,actual_vals,label='Actual values',color='black')
            ax.plot(x_range,fitted_vals,label='Fitted values',color='red')
            ax.set_title('Actual vs Fitted Plot')
            plt.plot()

    def get_fit_interval(self,alpha=0.05):
        if self.m is not None:
            r = self.m.predict_in_sample(level=[100-alpha*100])
            results = pd.DataFrame()
            results['lower_bound']=r[f'fitted-lo-{100-alpha*100}']
            results['upper_bound']=r[f'fitted-hi-{100-alpha*100}']
            return results
        else:
            raise NoConfidenceIntervalModelAbsent
    
    def get_forecast_interval(self,steps=5,alpha=0.05):
        if self.m is not None:
            r = self.m.predict(h=steps,level=[100-alpha*100])
            results = pd.DataFrame()
            results['lower_bound']=r[f'lo-{100-alpha*100}']
            results['upper_bound']=r[f'hi-{100-alpha*100}']
            return results
        else:
            raise NoConfidenceIntervalModelAbsent
        
class IMAPAModel(TSModel):
    def __init__(self,**kwargs):
        self.ts=None
        self.fitted_values=None 
        self.fit_options=None 
        self.model_init_opts=kwargs 
        self.m=IMAPA(**kwargs)
    
    def fit(self,ts,**kwargs):
        self.ts=ts 
        self.fit_options=kwargs
        self.m.fit(ts.values,**kwargs)
        self.fitted_values=self.m.predict_in_sample()['fitted']
        return self 
    def forecast(self,steps=5):
        if self.m is not None:
            frsct=self.m.predict(h=steps)['mean']
        else:
            raise NoForecastModelAbsent
        return frsct
    def summary(self):
        raise NotImplementedError
    
    def residual_analysis(self):
        if self.m is not None and self.ts is not None:
            r = self.ts-self.fitted_values
            fig1 = sm.qqplot(r,line='45')
            fig1.suptitle('QQ plot')
        else:
            raise NoDiagnosticsModelAbsent       

    def plot_fit(self):
        if self.m is not None and self.ts is not None:
            x_range = self.ts.index
            fitted_vals = self.fitted_values
            actual_vals = self.ts
            _,ax = plt.subplots(figsize=(5,5))
            ax.plot(x_range,actual_vals,label='Actual values',color='black')
            ax.plot(x_range,fitted_vals,label='Fitted values',color='red')
            ax.set_title('Actual vs Fitted Plot')
            plt.plot()
    def get_fit_interval(self,alpha=0.05):
        if self.m is not None:
            r = self.m.predict_in_sample(level=[100-alpha*100])
            results = pd.DataFrame()
            results['lower_bound']=r[f'fitted-lo-{100-alpha*100}']
            results['upper_bound']=r[f'fitted-hi-{100-alpha*100}']
            return results
        else:
            raise NoConfidenceIntervalModelAbsent
    
    def get_forecast_interval(self,steps=5,alpha=0.05):
        if self.m is not None:
            r = self.m.predict(h=steps,level=[100-alpha*100])
            results = pd.DataFrame()
            results['lower_bound']=r[f'lo-{100-alpha*100}']
            results['upper_bound']=r[f'hi-{100-alpha*100}']
            return results
        else:
            raise NoConfidenceIntervalModelAbsent