from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class LimitAlgorithm(BaseAlgorithm):
    def __init__(self, upper_limit=20, lower_limit=50):
        super().__init__()

        self._upper_limit = upper_limit
        self._lower_limit = lower_limit

    def get_confidence(self):
        pass

    def update(self, timestamp, value):
        self._samples = self._samples.append({'timestamp': timestamp, 'value': value}, ignore_index=True)

        if value < self._lower_limit:
            self._current_state = self.states.underutil_anomaly
            self._anomalies_underutil = self._anomalies_underutil.append({'timestamp': timestamp, 'value': value},
                                                                         ignore_index=True)
        elif value < self._upper_limit:
            self._current_state = self.states.normal
        else:
            self._current_state = self.states.overutil_anomaly
            self._anomalies_overutil = self._anomalies_overutil.append({'timestamp': timestamp, 'value': value},
                                                                       ignore_index=True)
