from statistics import stdev
from scipy.signal import savgol_filter

from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class SavgolAlgorithm(BaseAlgorithm):

    def __init__(self, window_length=7, poly_order=3, tolerance_multiplier=1, min_tolerance=1):
        super().__init__()

        self._window_length = window_length
        self._tolerance_multiplier = tolerance_multiplier
        self.min_tolerance = min_tolerance
        self._poly_order = poly_order

    def get_confidence(self):
        pass

    def update(self, timestamp, value):
        self._samples = self._samples.append({'timestamp': timestamp, 'value': value}, ignore_index=True)

        # recalculate normal state
        if len(self._samples['value']) < self._window_length:
            self._current_state = self.states.learning
            return

        self._normal_state = savgol_filter(self._samples.tail(self._window_length)['value'],
                                           window_length=self._window_length,
                                           polyorder=self._poly_order)[self._window_length//2]
        # the prediction at the ends changes based on neighbor values, therefore to get smoothed value its
        # necessary to measure value inside the window

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
        return "SavgolAlgorithm"
