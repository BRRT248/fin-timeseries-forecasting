
from statsmodels.tsa.arima.model import ARIMA

class ARIMAWrap:
    def __init__(self, order=(1,0,1)):
        self.order = order
        self.model_ = None

    def fit(self, y):
        self.model_ = ARIMA(y, order=self.order).fit()
        return self

    def predict_next(self):
        fc = self.model_.forecast(1)
        return float(fc.iloc[0])
