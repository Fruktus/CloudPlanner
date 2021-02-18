from abc import ABC
import pandas as pd

class BaseAlgorithm(ABC):
    def __init__(self):
        self.states = Enum('states', 'learning normal anomaly')

        self._samples = pd.DataFrame(columns = ['timestamp', 'value'])

    def get_stdev_tolerance(self):
        if len(self._samples) < 2:
            return 0
        return self._tolerance_multiplier * stdev(self._samples['value'])

    @abstractmethod
    def update(self, timestamp, value):
        pass