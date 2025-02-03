import warnings
from tslib.interface import Eval,TSModel
from tslib.models import Arima,ProphetModel,TBATSModel, CrostonModel, ADIDAModel, IMAPAModel
import pandas as pd
from tqdm import tqdm
warnings.simplefilter('ignore', category=UserWarning)
model_map = {'arima':Arima,'prophet':ProphetModel,'tbats':TBATSModel,'croston':CrostonModel,'adida':ADIDAModel,'imapa':IMAPAModel}

class CrossValidation(Eval):
    def evaluate(self,folds:list,model:TSModel,type:str,**kwargs)->dict:
        horizon=len(folds[0]['test'])
        results = []
        if model is None:
            raise Exception('''Provide a fitted model''')
        for idx,fold in tqdm(enumerate(folds),total=len(folds)):
            train=fold['train']
            if model.model_init_opts is not None:
                m = model_map[type](**model.model_init_opts)
            else:
                m = model_map[type]()
            if model.fit_options is not None:
                m = m.fit(train,**model.fit_options)
            else:
                m = m.fit(train)
            test=fold['test']
            frsct = m.forecast(steps=horizon,**kwargs)
            if not isinstance(frsct,pd.Series):
                frsct = pd.Series(frsct)
            res=pd.DataFrame({'actual':test.values,
                            'forecast':frsct.values,
                            'fold':idx,
                            'horizon':range(1,horizon+1)})
            results.append(res)
        raw_results = pd.concat(results)
        raw_results['abs_err'] = (raw_results['actual']-raw_results['forecast']).abs()
        raw_results['ape']=raw_results['abs_err']/raw_results['actual'].abs()
        raw_results['sape']=raw_results['abs_err']/(raw_results['actual'].abs()+raw_results['forecast'].abs())*0.5
        mae = raw_results.groupby('horizon')[['abs_err']].agg(['mean','std','min','max']).reset_index()
        mae.columns = ['horizon','mae','mae_std','mae_min','mae_max']
        mape = raw_results.groupby('horizon')[['ape']].agg(['mean','std','min','max']).reset_index()
        mape.columns = ['horizon','mape','mape_std','mape_min','mape_max']
        smape = raw_results.groupby('horizon')[['sape']].agg(['mean','std','min','max']).reset_index()
        smape.columns = ['horizon','smape','smape_std','smape_min','smape_max']
        return {'raw_results':raw_results,'mae':mae,'mape':mape,'smape':smape}

class AccuracyEval(Eval):
    def evaluate(self,forecast:pd.Series,actual:pd.Series):
        mape= ((forecast-actual)/actual).abs().mean()*100
        smape=((forecast-actual).abs().mean()/((forecast.abs()+actual.abs())/2).mean())*100
        mse = ((forecast-actual)**2).mean()
        result = {'mape':mape,'smape':smape,'mse':mse}
        return result

class ClumpinessEval(Eval):
    '''
    Uses the criteria laid out in:
    Syntetos, A. A., & Boylan, J. E. (2005). "The accuracy of intermittent demand estimates." International Journal of Forecasting, 21(2), 303-314.
    To understand the extent of clumpiness in the data.
    '''
    def evaluate(self,ts:pd.Series):
        tot_period = ts.shape[0]
        non_zero_period = ts[ts!=0].shape[0]
        if non_zero_period>0:
            adi = tot_period/non_zero_period
            non_zero_demand = ts[ts!=0]
            stdev_non_zero = non_zero_demand.std()
            mean_non_zero = non_zero_demand.mean()
            cv = stdev_non_zero/mean_non_zero
            demand_pattern = None
            if adi<=1.32 and cv<=0.49:
                demand_pattern='smooth: use standard forecast methods'
            elif adi<=1.32 and cv>0.49:
                demand_pattern='erratic: use robust methods to forecast'
            elif adi>1.32 and cv<=0.49:
                demand_pattern="intermitent: Use Croston's or TSB method"
            else:
                demand_pattern="lumpy: Try Croston, ADIDA, TSB, SBA, TBATS and ML methods"
            return {'adi':adi,'cv':cv,'demand_pattern':demand_pattern}
        else:
            raise Exception('No non zero period found, cant compute clumpiness')
        
        
        