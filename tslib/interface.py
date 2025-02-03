import abc
class TSModel(abc.ABC):
    def __init__(self):
        self.fit_options=None
        self.model_init_opts=None
    @abc.abstractmethod 
    def fit(self,data):
        raise NotImplementedError
    @abc.abstractmethod
    def forecast(self):
        raise NotImplementedError
    @abc.abstractmethod
    def residual_analysis(self):
        raise NotImplementedError
    @abc.abstractmethod
    def summary(self):
        raise NotImplementedError
    @abc.abstractmethod
    def plot_fit(self):
        raise NotImplementedError
    @abc.abstractmethod
    def get_fit_interval(self):
        raise NotImplementedError
    @abc.abstractmethod
    def get_forecast_interval(self):
        raise NotImplementedError

class Preprocessing(abc.ABC):
    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError

class Eval(abc.ABC):
    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError

class Report(abc.ABC):
    @abc.abstractmethod
    def add_model(self):
        raise NotImplemented
    @abc.abstractmethod
    def add_model_forecast(self):
        raise NotImplemented
    @abc.abstractmethod
    def add_cv_results(self):
        raise NotImplemented
    @abc.abstractmethod
    def report_model_performance(self):
        '''Would support both in sample and outsample'''
        raise NotImplemented
    @abc.abstractmethod
    def report_error_analysis(self):
        raise NotImplemented
    @abc.abstractmethod
    def report_cv(self):
        raise NotImplemented