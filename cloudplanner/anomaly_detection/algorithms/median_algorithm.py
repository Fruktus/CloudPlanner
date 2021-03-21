from statistics import median, stdev

from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class MedianAlgorithm(BaseAlgorithm):
    def __init__(self, store_last_n=7, tolerance_multiplier=2):
        super().__init__()

        self._tolerance_multiplier = tolerance_multiplier
        self._use_last_n = store_last_n

    def get_confidence(self):
        pass

    def update(self, timestamp, value):
        self._samples = self._samples.append({'timestamp': timestamp, 'value': value}, ignore_index=True)

        # recalculate normal state
        if len(self._samples['value']) < 2:
            self._current_state = self.states.learning
            return

        self._normal_state = median(self._samples.tail(self._use_last_n)['value'])

        tolerance = self._tolerance_multiplier * stdev(self._samples.tail(self._use_last_n)['value'])
        # TODO possibly calculate stdev over full history

        self._upper_treshold = self._normal_state + tolerance
        self._lower_treshold = self._normal_state - tolerance
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
        return "MedianAlgorithm"
