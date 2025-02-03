from interface import Report,TSModel
from evaluation import AccuracyEval
import pandas as pd

class ModelComparisonReport(Report):
    def __init__(self):
        self.models=[]
        self.forecasts=[]
        self.cv_results=[]
    def add_models(self,model,name):
        if not isinstance(model,TSModel):
            raise Exception('Model should be of type tslib')
        self.models.append({name:model})
        return self
    def add_model_forecast(self,frcst):
        if not isinstance(frcst,pd.Series):
            raise Exception("Forecast must be a pd.Series object")
        name=list(self.models[-1].items())[0][0]
        self.forecasts.append({name:frcst})
        return self
    def add_cv_results(self,res):
        if not isinstance(res,dict):
            raise Exception("Cross validation results should be dict")
        name=list(self.models[-1].items())[0][0]
        self.add_cv_results.append({name:res})
        return self
    
    def _report_model_performance_insample(self,model_name):
        results = []
        if model_name=='all':
            for model_dict in self.models:
                    for name in model_dict:
                        model = model_dict[name]
                        fitted_val = model.fitted_values
                        ts = model.ts
                        acc = AccuracyEval()
                        res = acc.evaluate(forecast=fitted_val,actual=ts)
                        res['model']=name
                        results.append(res)
        else:
            for model_dict in self.models:
                for name in model_dict:
                    if name==model_name:
                        model = model_dict[name]
                        fitted_val = model.fitted_values
                        ts = model.ts
                        acc = AccuracyEval()
                        res = acc.evaluate(forecast=fitted_val,actual=ts)
                        res['model']=name
                        results.append(res)
        return results

    def _report_model_performance_outsample(self,actual_f,model_name):
        results = []
        if model_name=='all':
                for model_dict in self.models:
                    for name in model_dict:
                        model = model_dict[name]
                        fitted_val = model.forecast(steps=len(actual_f))
                        ts = actual_f
                        acc = AccuracyEval()
                        res = acc.evaluate(forecast=fitted_val,actual=ts)
                        res['model']=name
                        results.append(res)
        else:
            for model_dict in self.models:
                for name in model_dict:
                    if name == model_name:
                        model = model_dict[name]
                        fitted_val = model.forecast(steps=len(actual_f))
                        ts = actual_f
                        acc = AccuracyEval()
                        res = acc.evaluate(forecast=fitted_val,actual=ts)
                        res['model']=name
                        results.append(res)
        return results
    def report_model_performance(self,actual_f,insample=True,model_name='all'):
        if insample:
            results = self._report_model_performance_insample(model_name)   
        else:
            results = self._report_model_performance_outsample(actual_f,model_name)
        return results
    def report_error_analysis(self,model_name='all'):
        if model_name=='all':
            for model_dict in self.models:
                for name in model_dict:
                    model_dict[name].residual_analysis()
        else:
            for model_dict in self.models:
                for name in model_dict:
                    if name==model_name:
                        model_dict[name].residual_analysis()
    def report_cv(self,model_name='all'):
        raise NotImplementedError

        
            




