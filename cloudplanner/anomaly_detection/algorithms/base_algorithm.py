from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
from statistics import stdev


class BaseAlgorithm(ABC):
    def __init__(self):
        self.states = Enum('states', 'learning normal overutil_anomaly underutil_anomaly')

        self._samples = pd.DataFrame(columns=['timestamp', 'value'])
        self._anomalies_overutil = pd.DataFrame(columns=['timestamp', 'value'])
        self._anomalies_underutil = pd.DataFrame(columns=['timestamp', 'value'])

        self._normal_state = 0
        self._current_state = self.states.learning

    def get_stdev(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return stdev(self._samples['value'])

    def get_anomaly_count(self, overutil_only=True) -> int:
        if overutil_only:
            return len(self._anomalies_overutil)
        return len(self._anomalies_overutil) + len(self._anomalies_underutil)

    def get_current_state(self) -> Enum:
        return self._current_state

    def get_history(self):
        return self._samples

    @abstractmethod
    def get_confidence(self):
        pass

    @abstractmethod
    def update(self, timestamp, value) -> None:
        pass

    @abstractmethod
    def __str__(self):
        pass
