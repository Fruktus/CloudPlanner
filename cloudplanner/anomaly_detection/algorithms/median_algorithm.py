import pandas as pd
from statistics import median, stdev

from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class MedianAlgorithm(BaseAlgorithm):
    def __init__(self, store_last_n=7, tolerance_multiplier=2):
        super().__init__()

        self._store_last_n = store_last_n
        self._last_n_samples = pd.DataFrame(columns=['timestamp', 'value'])

    def get_confidence(self):
        pass

    def update(self, timestamp, value):
        self._samples.append([timestamp, value])

        if len(self._last_n_samples) == self._store_last_n:
            self._last_n_samples.drop(self._last_n_samples.head().index, inplace=True)
        self._last_n_samples.append([timestamp, value])

        # recalculate normal state
        if len(self._last_n_samples['value']) < 2:
            self._current_state = self.states.learning
            return

        self._normal_state = median(self._last_n_samples['value'])

        # recalculate current state
        self._current_state = self.states.normal if value <= self._normal_state + self._normal_state * self._tolerance\
            else self.states.anomaly

        tolerance = self._tolerance_multiplier * stdev(self._last_n_samples['value'])
        # TODO possibly calculate stdev over full history

        if value < self._normal_state - tolerance:
            self._current_state = self.states.underutil_anomaly
            self._anomalies_underutil.append([timestamp, value])
        elif value < self._normal_state + tolerance:
            self._current_state = self.states.normal
        else:
            self._current_state = self.states.overutil_anomaly
            self._anomalies_overutil.append([timestamp, value])
