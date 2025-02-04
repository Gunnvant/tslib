## **tslib** wrapper library for commonly used time series forecasting algorithms

tslib provides a common interface to build a variety of time series forecasting models with the following methods:

1. `fit`
2. `forecast`
3. `residual_analysis`
4. `summary`
5. `plot_fit`
6. `get_fit_interval`
7. `get_forecast_interval`

It supports the following models:

| Model |Use Case  |
|--|--|
| ARIMA,ARIMAX and SARIAMX | Common time series scenarios involving stable single seasonality and exogenous variables |
| Prophet |Time series with multiple seasonality (weekly, yearly) and shock effects due to special events: holidays etc  |
|TBATS|Time series with multiple seasonality and sparse data|
| Croston, ADIDA, TBA, IMAPA | Handle time series with sparse data  |


The supported models are in `./tslib/models.py` file.

For evaluation and validation following classes are present:

1. CrossValidation: Useful to produce model accuracy analysis over different training lengths and evaluation time horizons.
2. CumpinessEval: Useful to find out the extent of clumpiness in time series data and suggest the models to be used.


This is still a work in progress and I plan to include the following:

1. Unit tests to automate CI pipelines
2. Functionality to use supervised methods such as RandomForest, Xgboost
3. Package this repo into a python package installable via pip, conda or uv.

You can find out how to use this library by referring to `./examples` folder

### How to get started:
This project is still a work in progress, so there is no python package. For the time being you can start using this library by setting up a conda environment using the `env.yaml` file.

Run the following command:

```shell
conda env create -f env.yaml
```

This will install the dependencies. You can keep the `tslib` folder in your working directory and follow along the notebooks in `examples` directory.