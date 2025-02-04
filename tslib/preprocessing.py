from tslib.interface import Preprocessing
from tslib.exceptions import NoCVDataSetPossible
import numpy as np
import pandas as pd

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

class TimeDelayEmbedding(Preprocessing):
    '''
    Ref: Cerqueira, Vitor, Nuno Moniz, and Carlos Soares. "VEST: Automatic Feature Engineering for Forecasting." arXiv preprint arXiv:2010.07137 (2020).
    https://github.com/vcerqueira/vest-python
    '''
    def __init__(self,ts:pd.Series,k=5):
        self.ts=ts 
        self.k=k 
    
    def fit(self):
        X,y = [],[]
        
        for i in range(len(self.ts)):
            # find the end of this pattern
            end_ix = i + self.k
            # check if we are beyond the sequence
            if end_ix > len(self.ts) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = self.ts.values[i:end_ix], self.ts.values[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)

        return X, y