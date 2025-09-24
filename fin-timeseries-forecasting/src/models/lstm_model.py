
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def set_seed(seed=42):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class LSTMWrap:
    def __init__(self, window=20, epochs=6, batch_size=32, lr=1e-3):
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.last_window_ = None

    def _build(self):
        model = keras.Sequential([
            layers.Input(shape=(self.window, 1)),
            layers.LSTM(32),
            layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse")
        return model

    def _to_supervised(self, r):
        X, y = [], []
        for i in range(len(r) - self.window):
            X.append(r[i:i+self.window])
            y.append(r[i+self.window])
        X = np.asarray(X).reshape(-1, self.window, 1)
        y = np.asarray(y).reshape(-1, 1)
        return X, y

    def fit(self, y):
        set_seed(42)
        r = np.asarray(y).astype(float)
        if len(r) <= self.window:
            # Edge case: fall back to mean
            self.model = None
            self.last_window_ = r[-self.window:] if len(r) >= self.window else r
            return self
        X, target = self._to_supervised(r)
        self.model = self._build()
        self.model.fit(X, target, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.last_window_ = r[-self.window:].reshape(1, self.window, 1)
        return self

    def predict_next(self):
        if self.model is None or self.last_window_ is None or len(self.last_window_) == 0:
            return 0.0
        pred = self.model.predict(self.last_window_, verbose=0)
        return float(pred.ravel()[0])
