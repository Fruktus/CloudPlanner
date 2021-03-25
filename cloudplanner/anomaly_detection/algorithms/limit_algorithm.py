from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class LimitAlgorithm(BaseAlgorithm):
    def __init__(self, upper_treshold=20, lower_treshold=50):
        super().__init__()

        self._upper_treshold = upper_treshold
        self._lower_treshold = lower_treshold

    def get_confidence(self):
        pass

    def update(self, timestamp, value):
        self._samples = self._samples.append({'timestamp': timestamp, 'value': value}, ignore_index=True)
        self._anomalies_treshold_history = self._anomalies_treshold_history.append(
            {'timestamp': timestamp,
             'upper_treshold': self._upper_treshold,
             'lower_treshold': self._lower_treshold},
            ignore_index=True)

        if value < self._lower_treshold:
            self._current_state = self.states.underutil_anomaly
            self._anomalies_underutil = self._anomalies_underutil.append({'timestamp': timestamp, 'value': value},
                                                                         ignore_index=True)
        elif value < self._upper_treshold:
            self._current_state = self.states.normal
        else:
            self._current_state = self.states.overutil_anomaly
            self._anomalies_overutil = self._anomalies_overutil.append({'timestamp': timestamp, 'value': value},
                                                                       ignore_index=True)

    def __str__(self):
        return "LimitAlgorithm"
