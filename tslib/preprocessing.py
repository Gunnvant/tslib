from tslib.interface import Preprocessing
from tslib.exceptions import NoCVDataSetPossible

class CVPreprocessor(Preprocessing):
    def __init__(self,start_idx,step_size,horizon):
        self.start_idx=start_idx
        self.step_size=step_size 
        self.horizon=horizon

    def fit(self,ts):
        folds = []
        if ts is not None:
            n_samples = len(ts)
            if n_samples-self.step_size-(self.start_idx+1)<=self.horizon:
                raise NoCVDataSetPossible
            start_idx =  self.start_idx
            while n_samples-self.step_size-(start_idx+1)>=self.horizon:
                train = ts[:start_idx]
                test = ts[start_idx:start_idx+self.horizon]
                start_idx+=self.step_size
                folds.append({'train':train,'test':test})
        return folds