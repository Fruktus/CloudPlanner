from abc import ABC, abstractmethod


class BaseNetworkModel(ABC):
    def __init__(self):
        pass

    def print_summary(self):
        print(self.model.summary())

    @abstractmethod
    def fit_model(self, x_train, y_train, verbose=True):
        pass

    def predict(self, data):
        return self.model.predict(data)
