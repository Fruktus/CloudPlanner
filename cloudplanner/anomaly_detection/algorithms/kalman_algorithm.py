from statistics import stdev
from pykalman import KalmanFilter

from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class KalmanAlgorithm(BaseAlgorithm):

    def __init__(self, window_length=7, tolerance_multiplier=1, min_tolerance=1):
        super().__init__()

        self._window_length = window_length
        self._tolerance_multiplier = tolerance_multiplier
        self.min_tolerance = min_tolerance
        self._kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

    def get_confidence(self):
        pass

    def update(self, timestamp, value):
        self._samples = self._samples.append({'timestamp': timestamp, 'value': value}, ignore_index=True)

        # recalculate normal state
        if len(self._samples['value']) < self._window_length:
            self._current_state = self.states.learning
            return

        self._normal_state = self._kf.smooth(self._samples.tail(self._window_length)['value'])[0][-1][0]
        # get the value of the prediction of last day

        tolerance = self._tolerance_multiplier * stdev(self._samples.tail(self._window_length)['value'])
        tolerance = tolerance if tolerance > self.min_tolerance else self.min_tolerance

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
        elif value <= self._upper_treshold:
            self._current_state = self.states.normal
        else:
            self._current_state = self.states.overutil_anomaly
            self._anomalies_overutil = self._anomalies_overutil.append({'timestamp': timestamp, 'value': value},
                                                                       ignore_index=True)

    def __str__(self):
        return "KalmanAlgorithm"
