from collections import defaultdict

from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class WeightedHybridAlgorithm(BaseAlgorithm):

    def __init__(self, filters: [(BaseAlgorithm, float)], min_confidence=0.8):
        super().__init__()

        self._filters = filters
        self._min_confidence = min_confidence
        self._recurrency_data = {'day_of_week': defaultdict(lambda: 0),
                                 'day_of_month': defaultdict(lambda: 0)}

    def get_confidence(self):
        pass

    def update(self, timestamp, value):
        self._samples = self._samples.append({'timestamp': timestamp, 'value': value}, ignore_index=True)

        combined_states = []
        for alg, confidence in self._filters:
            alg.update(timestamp, value)
            combined_states.append(alg.get_current_state())

        if self.states.learning in combined_states:
            self._current_state = self.states.learning
            return

        state_confidence = 0.0
        for i in range(len(combined_states)):
            tmp_state = 1 if combined_states[i] == self.states.overutil_anomaly else 0
            state_confidence += tmp_state * self._filters[i][1]

        state_confidence /= len(self._filters)
        self._update_recurrent(timestamp, state_confidence > self._min_confidence)

        if state_confidence >= self._min_confidence:
            if self._is_recurrent(timestamp):
                self._current_state = self.states.normal
            else:
                self._current_state = self.states.overutil_anomaly
                self._anomalies_overutil = self._anomalies_overutil.append({'timestamp': timestamp, 'value': value},
                                                                           ignore_index=True)
        else:
            self._current_state = self.states.normal

        self._anomalies_treshold_history = self._anomalies_treshold_history.append(
            {'timestamp': timestamp,
             'upper_treshold': self._upper_treshold,
             'lower_treshold': self._lower_treshold},
            ignore_index=True)

    def _update_recurrent(self, timestamp, is_anomaly: bool):
        if is_anomaly:
            self._recurrency_data['day_of_week'][timestamp.day_of_week] += 1
            self._recurrency_data['day_of_month'][timestamp.day] += 1
        else:
            dow = self._recurrency_data['day_of_week'][timestamp.day_of_week]
            self._recurrency_data['day_of_week'][timestamp.day_of_week] = dow - 1 if dow > 0 else 0

            dom = self._recurrency_data['day_of_month'][timestamp.day]
            self._recurrency_data['day_of_month'][timestamp.day] = dom - 1 if dom > 0 else 0

    def _is_recurrent(self, timestamp):
        return self._recurrency_data['day_of_week'][timestamp.day_of_week] > 2 or \
               self._recurrency_data['day_of_month'][timestamp.day] > 2

    def __str__(self):
        return "WeightedHybridAlgorithm"
